import pandas as pd
from pathlib import Path
from typing import Optional

REQUIRED_COLUMNS = {
    "Ticket Number": "ticket_number",
    "Title": "title",
    "Description": "description",
    "Account": "account",
    "Created": "created",
    "Issue Type": "issue_type",
}

OPTIONAL_COLUMNS = {
    "Resources": "resources",
    "Status": "status",
    "Total Hours Worked": "total_hours",
    "Billed Hours": "billed_hours",
    "Sub-Issue Type": "sub_issue_type",
}

# Ticket types that are not useful for noise analysis
EXCLUDED_ISSUE_TYPES = {"Email"}
EXCLUDED_SUB_ISSUE_TYPES = {"PHISH"}


def clean_dataframe(df: pd.DataFrame, exclude_noise_meta: bool = True) -> pd.DataFrame:
    """Clean a DataFrame containing Autotask tickets, injecting defaults for missing optional columns."""
    df.columns = [str(c).strip() for c in df.columns]

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Your Autotask export is missing these required columns: {missing_required}. Please add them to your Autotask Ticket Search view and export again.")

    # Inject missing optional columns with empty defaults
    for col_name in OPTIONAL_COLUMNS:
        if col_name not in df.columns:
            df[col_name] = ""

    # Rename all columns to internal snake_case names
    all_mapping = {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}
    df = df.rename(columns=all_mapping)

    df["created"] = pd.to_datetime(df["created"], errors="coerce")
    df["total_hours"] = pd.to_numeric(df["total_hours"], errors="coerce").fillna(0.0)
    df["billed_hours"] = pd.to_numeric(df["billed_hours"], errors="coerce").fillna(0.0)
    df["issue_type"] = df["issue_type"].fillna("").str.strip()
    df["sub_issue_type"] = df["sub_issue_type"].fillna("").astype(str).str.strip()
    df["account"] = df["account"].fillna("Unknown").str.strip()
    df["resources"] = df["resources"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["status"] = df["status"].fillna("").astype(str).str.strip()

    if exclude_noise_meta:
        df = df[~df["issue_type"].isin(EXCLUDED_ISSUE_TYPES)]
        df = df[~df["sub_issue_type"].isin(EXCLUDED_SUB_ISSUE_TYPES)]

    df = df.reset_index(drop=True)
    return df


def load_csv(path: str, exclude_noise_meta: bool = True) -> pd.DataFrame:
    """Load an Autotask ticket export CSV from disk and return a cleaned DataFrame."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return clean_dataframe(df, exclude_noise_meta)


def merge_csvs(paths: list[str], exclude_noise_meta: bool = True) -> pd.DataFrame:
    """Merge multiple CSV exports from disk, deduplicate by ticket number."""
    frames = [load_csv(p, exclude_noise_meta) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticket_number"], keep="last")
    return combined.reset_index(drop=True)
