from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Ticket:
    ticket_number: str
    title: str
    description: str
    account: str
    resources: str
    status: str
    created: datetime
    total_hours: float      # stored but not relied on for analysis — inaccurate in practice
    billed_hours: float
    sub_issue_type: str
    issue_type: str

    @property
    def short_description(self) -> str:
        return self.description[:300].strip() if self.description else ""


@dataclass
class Pattern:
    pattern_type: str           # recurring_issue | repeat_contact | same_day_burst
    account: str
    issue_type: str
    ticket_count: int
    tickets: list = field(default_factory=list)
    contact: Optional[str] = None
    sub_issue_breakdown: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    # reliable signals (do not depend on hours)
    unique_contacts: int = 0
    cluster_age_days: int = 0
    recurrence_rate: float = 0.0    # tickets per week within the cluster window
    account_noise_ratio: float = 0.0  # this cluster as % of account's total tickets


@dataclass
class Recommendation:
    pattern: Pattern
    pattern_summary: str
    root_cause: str
    recommendation_type: str    # automation | training | process_change | sales_opportunity
    recommended_action: str
    estimated_monthly_tickets_prevented: int
    source_ticket_numbers: list = field(default_factory=list)
