import json
import psycopg2
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from ..models.ticket import Pattern, Recommendation
from ..hatzai.client import HatzAIClient
from ..store.db import cache_get, cache_set, get_historical_context

SYSTEM_PROMPT = """\
You are a Noise Reduction analyst for TAG Solutions, a Managed Service Provider (MSP) in Albany, NY.

Your job is to analyze clusters of related IT support tickets and produce a structured Noise Reduction Committee (NRC) recommendation.

Noise = preventable reactive work. Your recommendations should identify what proactive action \
(automation, user training, process change, or new sales opportunity) would prevent these tickets from recurring.

Be specific and actionable. Reference ticket numbers and actual issue patterns. Keep language concise.
Note: ticket hour logs are not reliably maintained — do not estimate time savings. Focus on ticket volume reduction.
"""

# Only send clusters of this size or larger to the LLM — filters out trivial noise
MIN_LLM_TICKETS = 3

# Max concurrent LLM calls
LLM_CONCURRENCY = 5


def _select_representative_tickets(tickets: list, max_count: int = 15) -> list:
    """Pick the most informative tickets for the LLM prompt.

    Prioritises sub-issue-type variety first (full shape of the problem),
    then fills remaining slots with longest descriptions (most detail).
    """
    sorted_by_desc = sorted(
        tickets,
        key=lambda t: len(str(t.get("description", ""))),
        reverse=True,
    )

    seen_sub_issues: set = set()
    diverse: list = []
    remainder: list = []

    for t in sorted_by_desc:
        sub = t.get("sub_issue_type", "")
        if sub not in seen_sub_issues:
            diverse.append(t)
            seen_sub_issues.add(sub)
        else:
            remainder.append(t)
        if len(diverse) >= max_count:
            break

    slots_left = max_count - len(diverse)
    diverse.extend(remainder[:slots_left])
    return diverse[:max_count]


def _build_prompt(pattern: Pattern, hist_context: Optional[dict] = None) -> str:
    sample = _select_representative_tickets(pattern.tickets)

    ticket_lines = []
    for t in sample:
        desc = str(t.get("description", ""))[:250].replace("\n", " ")
        ticket_lines.append(
            f"  [{t['ticket_number']}] {t['title']} "
            f"| {t['issue_type']} / {t['sub_issue_type']} | {desc}"
        )

    tickets_block = "\n".join(ticket_lines)

    contact_line = (
        f"\nPrimary resource handling these tickets: {pattern.contact}"
        if pattern.contact else ""
    )
    extra_line = (
        f"\nAdditional context: {json.dumps(pattern.extra)}"
        if pattern.extra else ""
    )
    age_line = (
        f"\nDetection window span: {pattern.cluster_age_days} days "
        f"(first to most recent ticket)"
        if pattern.cluster_age_days > 0 else ""
    )

    hist_block = ""
    if hist_context:
        hist_block = (
            f"\nHistorical context (full store):"
            f"\n  All-time ticket count : {hist_context['all_time_count']}"
            f"\n  Pattern first seen    : {hist_context['first_seen'] or 'unknown'}"
            f"\n  Trend                 : {hist_context['trend_label']}"
        )

    return f"""\
Analyze the following cluster of support tickets and generate a Noise Reduction Committee recommendation.

Pattern type: {pattern.pattern_type}
Account: {pattern.account}
Issue type: {pattern.issue_type}
Ticket count in detection window: {pattern.ticket_count} \
({round(pattern.account_noise_ratio * 100, 1)}% of this account's tickets)
Unique contacts/resources involved: {pattern.unique_contacts}
Recurrence rate: ~{pattern.recurrence_rate} tickets/week
Sub-issue breakdown: {json.dumps(pattern.sub_issue_breakdown)}{contact_line}{age_line}{extra_line}{hist_block}

Representative sample ({len(sample)} of {pattern.ticket_count} tickets):
{tickets_block}

Respond with ONLY a JSON object in this exact schema (no code block tags, no explanation):
{{
  "pattern_summary": "<bullet points (use markdown '-' or '*') describing the recurring pattern>",
  "root_cause": "<bullet points explaining what is actually driving these tickets>",
  "recommendation_type": "<one of: automation | training | process_change | sales_opportunity>",
  "recommended_action": "<bullet points detailing specific, actionable steps TAG Solutions should take>",
  "estimated_monthly_tickets_prevented": <integer>
}}
"""


def _ticket_numbers(pattern: Pattern) -> list[str]:
    return [t["ticket_number"] for t in pattern.tickets]


def _result_to_recommendation(result: dict, pattern: Pattern) -> Recommendation:
    return Recommendation(
        pattern=pattern,
        pattern_summary=result.get("pattern_summary", ""),
        root_cause=result.get("root_cause", ""),
        recommendation_type=result.get("recommendation_type", "process_change"),
        recommended_action=result.get("recommended_action", ""),
        estimated_monthly_tickets_prevented=int(
            result.get("estimated_monthly_tickets_prevented", 0)
        ),
        source_ticket_numbers=_ticket_numbers(pattern),
    )


def generate_recommendations(
    patterns: List[Pattern],
    client: HatzAIClient,
    conn: Optional[psycopg2.extensions.connection] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force_refresh: bool = False,
    max_patterns: int = 20,
) -> List[Recommendation]:
    """Generate NRC recommendations with caching and concurrent LLM calls.

    Flow:
      1. Filter insignificant patterns (< MIN_LLM_TICKETS)
      2. Check cache for each pattern (sequential — fast DB reads)
      3. Fetch historical context for patterns that need LLM calls
      4. Run LLM calls concurrently (ThreadPoolExecutor)
      5. Write new results to cache (sequential — safe DB writes)
      6. Sort by estimated impact
    """
    candidates = [p for p in patterns[:max_patterns] if p.ticket_count >= MIN_LLM_TICKETS]
    filtered_out = len(patterns[:max_patterns]) - len(candidates)
    if filtered_out:
        print(f"  Skipping {filtered_out} pattern(s) below significance threshold "
              f"({MIN_LLM_TICKETS} tickets).")

    # ── phase 1: cache lookup ──────────────────────────────────────────────────
    needs_llm: list[Pattern] = []
    cached_recs: list[Recommendation] = []
    cache_hits = 0

    for pattern in candidates:
        ticket_nums = _ticket_numbers(pattern)
        cached = None
        if conn and not force_refresh:
            cached = cache_get(
                pattern.pattern_type, pattern.account, pattern.issue_type,
                ticket_nums, client.model, conn,
            )
        if cached:
            cache_hits += 1
            print(f"  [cached]  {pattern.pattern_type} | {pattern.account} "
                  f"| {pattern.ticket_count} tickets")
            cached_recs.append(_result_to_recommendation(cached, pattern))
        else:
            needs_llm.append(pattern)

    # ── phase 2: historical context for LLM patterns ──────────────────────────
    hist_contexts: dict[str, dict] = {}
    if conn and start_date and end_date and needs_llm:
        for pattern in needs_llm:
            key = f"{pattern.account}||{pattern.issue_type}"
            if key not in hist_contexts:
                hist_contexts[key] = get_historical_context(
                    pattern.account, pattern.issue_type, start_date, end_date, conn
                )

    # ── phase 3: concurrent LLM calls ─────────────────────────────────────────
    llm_results: list[tuple[Pattern, dict]] = []
    print_lock = threading.Lock()

    def call_llm(pattern: Pattern) -> tuple[Pattern, dict]:
        hist = hist_contexts.get(f"{pattern.account}||{pattern.issue_type}")
        with print_lock:
            print(f"  [llm]     {pattern.pattern_type} | {pattern.account} "
                  f"| {pattern.ticket_count} tickets...")
        result = client.chat_json(_build_prompt(pattern, hist), system=SYSTEM_PROMPT)
        return pattern, result

    if needs_llm:
        with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
            futures = {executor.submit(call_llm, p): p for p in needs_llm}
            for future in as_completed(futures):
                pattern = futures[future]
                try:
                    _, result = future.result()
                    llm_results.append((pattern, result))
                except Exception as e:
                    with print_lock:
                        print(f"  WARNING: failed for ({pattern.account} / "
                              f"{pattern.issue_type}): {e}")

    # ── phase 4: write new results to cache ────────────────────────────────────
    new_recs: list[Recommendation] = []
    if conn:
        for pattern, result in llm_results:
            cache_set(
                pattern.pattern_type, pattern.account, pattern.issue_type,
                _ticket_numbers(pattern), client.model, result, conn,
            )
    for pattern, result in llm_results:
        new_recs.append(_result_to_recommendation(result, pattern))

    # ── summary line ───────────────────────────────────────────────────────────
    if conn:
        print(f"\n  Cache: {cache_hits} hits, {len(llm_results)} LLM calls made.")

    all_recs = cached_recs + new_recs
    all_recs.sort(key=lambda r: -r.estimated_monthly_tickets_prevented)
    return all_recs
