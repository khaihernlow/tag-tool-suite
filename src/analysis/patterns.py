import pandas as pd
from typing import List
from ..models.ticket import Pattern

MIN_CLUSTER_SIZE = 2


DEDUP_OVERLAP_THRESHOLD = 0.6  # skip a pattern if >60% of its tickets already appear in an accepted pattern


def find_patterns(df: pd.DataFrame) -> List[Pattern]:
    # pre-compute per-account totals once so every detector can compute noise ratio
    account_totals = df.groupby("account").size().to_dict()

    patterns: List[Pattern] = []
    patterns.extend(_recurring_issue_patterns(df, account_totals))
    patterns.extend(_repeat_contact_patterns(df, account_totals))
    patterns.extend(_same_day_burst_patterns(df, account_totals))

    patterns.sort(key=lambda p: (p.ticket_count, p.unique_contacts), reverse=True)
    return _deduplicate(patterns)


def _deduplicate(patterns: List[Pattern]) -> List[Pattern]:
    """Remove patterns whose ticket set substantially overlaps with an already-accepted pattern."""
    accepted: List[Pattern] = []
    accepted_tickets: List[set] = []

    for pattern in patterns:
        tickets = {t["ticket_number"] for t in pattern.tickets}
        duplicate = False
        for seen in accepted_tickets:
            overlap = len(tickets & seen) / len(tickets)
            if overlap >= DEDUP_OVERLAP_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            accepted.append(pattern)
            accepted_tickets.append(tickets)

    return accepted


def _to_records(group: pd.DataFrame) -> list:
    cols = ["ticket_number", "title", "description", "resources",
            "created", "issue_type", "sub_issue_type"]
    return group[cols].to_dict("records")


def _pattern_stats(group: pd.DataFrame, account: str, account_totals: dict) -> dict:
    """Compute reliable, hours-free stats for a cluster."""
    ticket_count = len(group)
    unique_contacts = group["resources"].nunique()

    if ticket_count > 1:
        age_days = (group["created"].max() - group["created"].min()).days
    else:
        age_days = 0

    # minimum 1 day to avoid division by zero; same-day bursts correctly get a high rate
    weeks = max(age_days / 7, 1 / 7)
    recurrence_rate = round(ticket_count / weeks, 2)

    noise_ratio = round(ticket_count / max(account_totals.get(account, 1), 1), 3)

    return {
        "unique_contacts": unique_contacts,
        "cluster_age_days": age_days,
        "recurrence_rate": recurrence_rate,
        "account_noise_ratio": noise_ratio,
    }


def _recurring_issue_patterns(df: pd.DataFrame, account_totals: dict) -> List[Pattern]:
    """Same account + issue_type appearing multiple times."""
    results = []
    valid = df[df["issue_type"] != ""]
    for (account, issue_type), group in valid.groupby(["account", "issue_type"]):
        if len(group) < MIN_CLUSTER_SIZE:
            continue
        stats = _pattern_stats(group, account, account_totals)
        results.append(Pattern(
            pattern_type="recurring_issue",
            account=account,
            issue_type=issue_type,
            ticket_count=len(group),
            tickets=_to_records(group),
            sub_issue_breakdown=group["sub_issue_type"].value_counts().to_dict(),
            **stats,
        ))
    return results


def _repeat_contact_patterns(df: pd.DataFrame, account_totals: dict) -> List[Pattern]:
    """Single technician resource handling many tickets for the same account."""
    results = []

    def extract_primary(resources_str: str) -> str:
        if not resources_str:
            return ""
        for part in resources_str.split(";"):
            if "(primary)" in part.lower():
                return part.replace("(primary)", "").strip()
        return resources_str.split(";")[0].strip()

    df = df.copy()
    df["_primary_resource"] = df["resources"].apply(extract_primary)

    valid = df[df["_primary_resource"] != ""]
    for (account, contact), group in valid.groupby(["account", "_primary_resource"]):
        if len(group) < MIN_CLUSTER_SIZE:
            continue
        stats = _pattern_stats(group, account, account_totals)
        results.append(Pattern(
            pattern_type="repeat_contact",
            account=account,
            issue_type="(multiple)",
            ticket_count=len(group),
            tickets=_to_records(group),
            contact=contact,
            sub_issue_breakdown=group["issue_type"].value_counts().to_dict(),
            **stats,
        ))
    return results


def _same_day_burst_patterns(df: pd.DataFrame, account_totals: dict) -> List[Pattern]:
    """3+ tickets from the same account on the same calendar day."""
    df = df.copy()
    df["_date"] = df["created"].dt.date
    results = []
    for (account, date), group in df.groupby(["account", "_date"]):
        if len(group) < 3:
            continue
        stats = _pattern_stats(group, account, account_totals)
        results.append(Pattern(
            pattern_type="same_day_burst",
            account=account,
            issue_type="(multiple)",
            ticket_count=len(group),
            tickets=_to_records(group),
            sub_issue_breakdown=group["issue_type"].value_counts().to_dict(),
            extra={"date": str(date)},
            **stats,
        ))
    return results
