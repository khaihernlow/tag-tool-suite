"""
NRC AI -- Noise Reduction Committee Analysis Tool
Tag Solutions, Albany NY

Commands:
  python main.py import <csv_file> [...]            Import tickets into local store
  python main.py analyze                             Analyze all stored tickets
  python main.py analyze --window 30                 Analyze last 30 days only
  python main.py analyze --since 2026-01-01          Analyze tickets since a date
  python main.py analyze <csv_file> [...]            Import then analyze in one step
  python main.py analyze --force-refresh             Re-run LLM on all patterns (ignore cache)
  python main.py status                              Show store and cache stats
  python main.py models                              List available HatzAI models
  python main.py cache-clear                         Wipe cached recommendations
  python main.py serve                               Start the web interface (default: 0.0.0.0:8000)
  python main.py serve --host 127.0.0.1 --port 5000 Start on a specific host/port
  python main.py serve --dev                         Enable auto-reload for development
"""

import argparse
import sys
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

from src.hatzai.client import HatzAIClient, HatzAIError
from src.ingest.csv_importer import load_csv
from src.analysis.patterns import find_patterns
from src.analysis.recommender import generate_recommendations
from src.store.db import (
    connect, import_tickets, load_tickets,
    ticket_count, cache_clear, cache_stats,
    since_date_from_window,
)


# ── display helpers ────────────────────────────────────────────────────────────

PRIORITY_LABEL = {"high": "*** HIGH ***", "medium": "  MEDIUM   ", "low": "   low    "}
REC_TYPE_LABEL = {
    "automation": "Automation",
    "training": "User Training",
    "process_change": "Process Change",
    "sales_opportunity": "Sales Opportunity",
}


def print_header(text: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_models(_args) -> None:
    client = HatzAIClient()
    print("Fetching available models from HatzAI...")
    models = client.list_models()
    if not models:
        print("No models returned.")
        return
    rows = [
        [m.get("name", "?"), m.get("display_name", ""), m.get("developer", "")]
        for m in models if isinstance(m, dict)
    ]
    print(tabulate(rows, headers=["Model ID (use in .env)", "Display Name", "Developer"], tablefmt="simple"))


def cmd_status(_args) -> None:
    conn = connect()
    total = ticket_count(conn)
    stats = cache_stats(conn)
    conn.close()
    print_header("STORE STATUS")
    print(f"  Tickets in store         : {total}")
    print(f"  Cached recommendations   : {stats['cached_recommendations']}")
    print()


def cmd_serve(args) -> None:
    import uvicorn
    print(f"Starting NRC AI web interface on http://{args.host}:{args.port}")
    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=args.dev,
    )


def cmd_cache_clear(_args) -> None:
    conn = connect()
    deleted = cache_clear(conn)
    conn.close()
    print(f"Cleared {deleted} cached recommendation(s).")


def cmd_import(args) -> None:
    """Import one or more CSV files into the persistent store."""
    conn = connect()
    total_new = 0
    total_skipped = 0

    for path in args.csv_files:
        print(f"Loading {path}...")
        try:
            df = load_csv(path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        new, skipped = import_tickets(df, conn)
        total_new += new
        total_skipped += skipped
        print(f"  {new} new tickets imported, {skipped} duplicates skipped.")

    conn.close()
    print(f"\nDone. Store now contains {ticket_count(connect())} tickets total.")


def cmd_analyze(args) -> None:
    conn = connect()

    # ── resolve detection window ───────────────────────────────────────────────
    since_date = None
    window_label = "all time"
    if getattr(args, "since", None):
        since_date = args.since
        window_label = f"since {since_date}"
    elif getattr(args, "window", None):
        since_date = since_date_from_window(args.window)
        window_label = f"last {args.window} days (since {since_date})"

    # ── optionally import CSV files first ──────────────────────────────────────
    if hasattr(args, "csv_files") and args.csv_files:
        print(f"Importing {len(args.csv_files)} CSV file(s)...")
        for path in args.csv_files:
            try:
                df = load_csv(path)
                new, skipped = import_tickets(df, conn)
                print(f"  {path}: {new} new, {skipped} duplicates skipped.")
            except Exception as e:
                print(f"  ERROR loading {path}: {e}")

    # ── load tickets (detection scope) ────────────────────────────────────────
    df = load_tickets(conn, since_date=since_date)
    if df.empty:
        print("\nNo tickets in store. Run: python main.py import <csv_file>")
        conn.close()
        return

    print(f"\nAnalyzing {len(df)} tickets across {df['account'].nunique()} accounts "
          f"({window_label}).")

    # ── summary ────────────────────────────────────────────────────────────────
    print_header("TICKET SUMMARY")
    account_counts = df.groupby("account").size().sort_values(ascending=False).head(10)
    print(tabulate(
        [[a, c] for a, c in account_counts.items()],
        headers=["Account", "Tickets"], tablefmt="simple",
    ))
    print()
    issue_counts = df.groupby("issue_type").size().sort_values(ascending=False).head(10)
    print(tabulate(
        [[i, c] for i, c in issue_counts.items()],
        headers=["Issue Type", "Count"], tablefmt="simple",
    ))

    # ── pattern detection ──────────────────────────────────────────────────────
    print_header("DETECTED PATTERNS")
    patterns = find_patterns(df)

    if not patterns:
        print("No patterns detected. More ticket data may be needed.")
        conn.close()
        return

    top_n = getattr(args, "top", 20) or 20
    patterns = patterns[:top_n]

    print(tabulate(
        [
            [i, p.pattern_type, p.account[:35], p.issue_type[:25],
             p.ticket_count, p.unique_contacts,
             f"{p.recurrence_rate}/wk", f"{round(p.account_noise_ratio * 100, 1)}%"]
            for i, p in enumerate(patterns, 1)
        ],
        headers=["#", "Pattern Type", "Account", "Issue Type",
                 "Tickets", "Contacts", "Rate", "% of Acct"],
        tablefmt="simple",
    ))

    if getattr(args, "no_llm", False):
        print("\n(LLM analysis skipped -- run without --no-llm to generate recommendations)")
        conn.close()
        return

    # ── recommendations ────────────────────────────────────────────────────────
    force = getattr(args, "force_refresh", False)
    client = HatzAIClient()

    print_header(f"GENERATING NRC RECOMMENDATIONS (model: {client.model})")
    if force:
        print("  --force-refresh: cache bypassed, all patterns will be re-analyzed.\n")

    try:
        recommendations = generate_recommendations(
            patterns, client, conn=conn, since_date=since_date, force_refresh=force
        )
    except HatzAIError as e:
        print(f"ERROR calling HatzAI: {e}")
        conn.close()
        sys.exit(1)

    conn.close()

    if not recommendations:
        print("No recommendations generated.")
        return

    print_header(f"NRC RECOMMENDATIONS ({len(recommendations)} total)")

    for i, rec in enumerate(recommendations, 1):
        priority_label = PRIORITY_LABEL.get(rec.priority, rec.priority).upper()
        type_label = REC_TYPE_LABEL.get(rec.recommendation_type, rec.recommendation_type)

        print(f"\n{'-'*78}")
        print(f"  #{i}  [{priority_label}]  {type_label}")
        print(f"  Account : {rec.pattern.account}")
        print(f"  Tickets : {', '.join(rec.source_ticket_numbers[:8])}"
              + (" ..." if len(rec.source_ticket_numbers) > 8 else ""))
        print(f"  Impact  : ~{rec.estimated_monthly_tickets_prevented} tickets/mo prevented")
        print()
        print(f"  PATTERN\n  {rec.pattern_summary}")
        print()
        print(f"  ROOT CAUSE\n  {rec.root_cause}")
        print()
        print(f"  RECOMMENDED ACTION\n  {rec.recommended_action}")

    print(f"\n{'-'*78}")
    total_tickets = sum(r.estimated_monthly_tickets_prevented for r in recommendations)
    print(f"\n  TOTAL ESTIMATED IMPACT (monthly)")
    print(f"  Tickets prevented : ~{total_tickets}")
    print()


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NRC AI -- Noise Reduction Committee Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # models
    sub.add_parser("models", help="List available HatzAI models")

    # status
    sub.add_parser("status", help="Show ticket store and cache stats")

    # cache-clear
    sub.add_parser("cache-clear", help="Wipe all cached LLM recommendations")

    # import
    imp = sub.add_parser("import", help="Import ticket CSVs into the store")
    imp.add_argument("csv_files", nargs="+", metavar="CSV_FILE")

    # serve
    srv = sub.add_parser("serve", help="Start the web interface")
    srv.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    srv.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    srv.add_argument("--dev", action="store_true", help="Enable auto-reload (development mode)")

    # analyze
    ana = sub.add_parser("analyze", help="Analyze stored tickets and generate recommendations")
    ana.add_argument("csv_files", nargs="*", metavar="CSV_FILE",
                     help="Optional: import these CSVs before analyzing")
    ana.add_argument("--top", type=int, default=20, metavar="N",
                     help="Max patterns to analyze (default 20)")
    ana.add_argument("--no-llm", action="store_true",
                     help="Show pattern stats only, skip LLM calls")
    ana.add_argument("--force-refresh", action="store_true",
                     help="Ignore recommendation cache, re-run LLM on all patterns")

    window_group = ana.add_mutually_exclusive_group()
    window_group.add_argument("--window", type=int, metavar="DAYS",
                              help="Only analyze tickets from the last N days "
                                   "(historical data still used for trend context)")
    window_group.add_argument("--since", metavar="DATE",
                              help="Only analyze tickets on or after this date "
                                   "(ISO format: YYYY-MM-DD)")

    args = parser.parse_args()

    if args.command == "models":
        cmd_models(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "cache-clear":
        cmd_cache_clear(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
