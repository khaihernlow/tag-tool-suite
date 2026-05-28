import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

load_dotenv()

from src.store.db import (
    connect, import_tickets, load_tickets,
    ticket_stats, cache_clear, since_date_from_window,
)
from src.ingest.csv_importer import load_csv
from src.analysis.patterns import find_patterns
from src.analysis.recommender import generate_recommendations
from src.hatzai.client import HatzAIClient

BASE_DIR = Path(__file__).parent
app = FastAPI(title="NRC AI")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── shared job state (one analysis at a time) ─────────────────────────────────

_job: dict = {
    "status": "idle",   # idle | running | done | error
    "log": [],
    "recommendations": [],
    "error": None,
}


from typing import Any

class _LogCapture:
    """Tee sys.stdout to both the terminal and the in-memory job log."""

    def __init__(self, original_stream: Any) -> None:
        self._original = original_stream
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        if hasattr(self._original, "write"):
            self._original.write(text)
        self._buf += text
        if "\n" in self._buf:
            parts = self._buf.split("\n")
            self._buf = parts[-1]
            with self._lock:
                for line in parts[:-1]:
                    if line.strip():
                        _job["log"].append(line)

    def flush(self) -> None:
        if hasattr(self._original, "flush"):
            self._original.flush()
        if self._buf.strip():
            with self._lock:
                _job["log"].append(self._buf)
            self._buf = ""


def _run_analysis(options: dict) -> None:
    """Background thread: full analysis pipeline."""
    original_stdout = sys.stdout
    capture = _LogCapture(original_stdout)
    sys.stdout = capture
    conn = None

    try:
        conn = connect()
        start_date: Optional[str] = options.get("start_date")
        end_date: Optional[str] = options.get("end_date")

        df = load_tickets(conn, start_date=start_date, end_date=end_date)
        if df.empty:
            print("No tickets in store. Import CSV files first.")
            _job["status"] = "done"
            return

        window_desc = f"from {start_date[:10]} to {end_date[:10]}" if (start_date and end_date) else "all time"
        print(f"Loaded {len(df)} tickets ({window_desc}) "
              f"across {df['account'].nunique()} accounts.")

        patterns = find_patterns(df)
        if not patterns:
            print("No patterns detected — more ticket data may be needed.")
            _job["status"] = "done"
            return

        top_n = options.get("top", 20)
        patterns = patterns[:top_n]
        print(f"Detected {len(patterns)} pattern(s). Starting LLM analysis...")

        if options.get("no_llm"):
            print("LLM analysis skipped (no-llm mode).")
            _job["status"] = "done"
            return

        client = HatzAIClient()
        recs = generate_recommendations(
            patterns,
            client,
            conn=conn,
            start_date=start_date,
            end_date=end_date,
            force_refresh=options.get("force_refresh", False),
        )

        _job["recommendations"] = [_rec_to_dict(r) for r in recs]
        total = sum(r.estimated_monthly_tickets_prevented for r in recs)
        print(f"\nDone. {len(recs)} recommendation(s) | "
              f"~{total} tickets/mo estimated impact.")
        _job["status"] = "done"

    except Exception as exc:
        _job["error"] = str(exc)
        _job["status"] = "error"
        print(f"ERROR: {exc}")
    finally:
        sys.stdout = original_stdout
        if conn:
            conn.close()


def _rec_to_dict(rec) -> dict:
    return {
        "recommendation_type": rec.recommendation_type,
        "account": rec.pattern.account,
        "pattern_type": rec.pattern.pattern_type,
        "ticket_count": rec.pattern.ticket_count,
        "recurrence_rate": rec.pattern.recurrence_rate,
        "account_noise_ratio": round(rec.pattern.account_noise_ratio * 100, 1),
        "unique_contacts": rec.pattern.unique_contacts,
        "source_ticket_numbers": rec.source_ticket_numbers,
        "estimated_monthly_tickets_prevented": rec.estimated_monthly_tickets_prevented,
        "pattern_summary": rec.pattern_summary,
        "root_cause": rec.root_cause,
        "recommended_action": rec.recommended_action,
    }


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/logo.png")
async def get_logo():
    return FileResponse(str(Path(__file__).parent / "logo.png"))

@app.get("/")
async def dashboard(request: Request, message: str = ""):
    conn = connect()
    stats = ticket_stats(conn)
    conn.close()
    return templates.TemplateResponse(request, "index.html", {
        "message": message,
        "job": _job,
        **stats,
    })


@app.post("/import")
async def handle_import(files: list[UploadFile] = File(...)):
    tmp_dir = Path(tempfile.mkdtemp())
    total_new = total_skipped = 0
    errors: list[str] = []
    stats = {"ticket_count": 0, "min_date": "-", "max_date": "-"}

    try:
        conn = connect()
        for upload in files:
            tmp_path = tmp_dir / (upload.filename or "upload.csv")
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            try:
                df = load_csv(str(tmp_path))
                new, skipped = import_tickets(df, conn)
                total_new += new
                total_skipped += skipped
            except Exception as exc:
                errors.append(f"{upload.filename}: {exc}")
        stats = ticket_stats(conn)
        conn.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if errors:
        return JSONResponse({
            "success": False,
            "error": "; ".join(errors),
            "ticket_count": stats["ticket_count"],
            "min_date": stats["min_date"],
            "max_date": stats["max_date"]
        }, status_code=400)

    return JSONResponse({
        "success": True,
        "message": f"{total_new} new ticket(s) imported, {total_skipped} duplicate(s) skipped.",
        "ticket_count": stats["ticket_count"],
        "min_date": stats["min_date"],
        "max_date": stats["max_date"]
    })


@app.post("/analyze/start")
async def start_analysis(request: Request):
    if _job["status"] == "running":
        return JSONResponse({"error": "An analysis is already running."}, status_code=409)

    form = await request.form()

    window_str = str(form.get("window", "")).strip()
    start_str = str(form.get("start_date", "")).strip()
    end_str = str(form.get("end_date", "")).strip()
    top_str = str(form.get("top", "20")).strip()
    force_refresh = "force_refresh" in form
    no_llm = "no_llm" in form

    window = int(window_str) if window_str.isdigit() else None
    top = int(top_str) if top_str.isdigit() else 20

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    if start_str:
        start_date = start_str
        if end_str:
            end_date = end_str + "T23:59:59"
    elif window:
        start_date = since_date_from_window(window)
        from datetime import date
        end_date = date.today().isoformat() + "T23:59:59"

    _job.update({
        "status": "running",
        "log": [],
        "recommendations": [],
        "error": None,
    })

    thread = threading.Thread(
        target=_run_analysis,
        args=({"start_date": start_date, "end_date": end_date, "top": top,
               "force_refresh": force_refresh, "no_llm": no_llm},),
        daemon=True,
    )
    thread.start()

    if os.environ.get("VERCEL"):
        import asyncio
        while thread.is_alive():
            await asyncio.sleep(0.5)
        return JSONResponse({
            "status": _job["status"],
            "log": _job["log"],
            "recommendations": _job["recommendations"],
            "error": _job["error"],
        })

    return JSONResponse({"status": "started"})



@app.get("/analyze/status")
async def analysis_status():
    return JSONResponse({
        "status": _job["status"],
        "log": _job["log"],
        "recommendations": _job["recommendations"],
        "error": _job["error"],
    })


@app.post("/cache-clear")
async def handle_cache_clear():
    conn = connect()
    deleted = cache_clear(conn)
    stats = ticket_stats(conn)
    conn.close()
    return JSONResponse({
        "success": True,
        "message": f"Cleared {deleted} cached recommendation(s).",
        "ticket_count": stats["ticket_count"],
        "min_date": stats["min_date"],
        "max_date": stats["max_date"]
    })
