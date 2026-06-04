# TAG Tool Suite

A centralized portal for internal AI and automation tools built for **TAG Solutions** (Albany, NY). 

The platform is designed as an easily extensible web dashboard, protected by Microsoft Azure AD Single Sign-On (SSO). It provides employees with instant access to custom-built business solutions without requiring them to install CLI scripts or run Python locally.

---

## Current Tools

### 1. NRC AI (Noise Reduction Committee Analysis Engine)
An AI-assisted ticket analysis tool that ingests IT support ticket data exported from Autotask, detects recurring patterns, and uses the HatzAI LLM API to generate actionable recommendations. It identifies what proactive steps (automation, training, process changes, or sales opportunities) would prevent tickets from recurring.

*See the [NRC AI Deep Dive](#nrc-ai-deep-dive) section below for technical details on how the clustering and LLM prompts work.*

---

## Architecture

The platform runs entirely as a serverless web application.

- **Frontend/Backend:** Python FastAPI with Jinja2 HTML templates.
- **Authentication:** OAuth 2.0 integration with Microsoft Entra ID (Azure AD). Validates `@tagsolutions.com` credentials.
- **Deployment:** Vercel (Serverless Python Functions).
- **Database (Local):** SQLite (`data/nrc.db`) is used locally for caching and analysis. 

### Project Structure

```
tag-tool-suite/
├── api/
│   └── index.py                    Vercel serverless entry point
├── web/
│   ├── app.py                      FastAPI application and route definitions
│   ├── templates/                  Jinja2 HTML templates (auth, hub, tools)
│   └── static/                     (Favicon, logos)
├── src/
│   ├── models/                     Data models
│   ├── ingest/                     Autotask CSV importer
│   ├── store/                      SQLite database interaction
│   ├── hatzai/                     HatzAI LLM API Client
│   └── analysis/                   Pattern detection and LLM recommender
├── main.py                         Legacy CLI entry point (optional)
├── requirements.txt                Python dependencies
└── vercel.json                     Vercel deployment configuration
```

---

## Setup & Local Development

**Requirements:** Python 3.12+

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your credentials:
   ```env
   # HatzAI Configuration
   HATZAI_API_KEY=your-hatzai-api-key
   HATZAI_MODEL=anthropic.claude-sonnet-4-6

   # Microsoft Azure AD SSO Configuration
   AZURE_CLIENT_ID=your-azure-client-id
   AZURE_CLIENT_SECRET=your-azure-client-secret
   AZURE_TENANT_ID=your-azure-tenant-id
   SECRET_KEY=a-long-random-string-for-session-cookies
   ```

3. Run the local development server:
   ```bash
   uvicorn web.app:app --reload --port 8000
   ```

4. Visit `http://localhost:8000` in your browser.

---

## Deployment (Vercel)

The application is deployed to **Vercel** via seamless GitHub integration. 
Every push to the `main` branch automatically triggers a new production build.

**Production URL:** `https://tools.tagsolutions.com`

**Environment Variables:**
The `.env` file is intentionally ignored by `.gitignore`. For the production site to work, the 4 `AZURE_*` and `SECRET_KEY` variables must be manually added to the **Vercel Dashboard** -> Project Settings -> Environment Variables.

---

## NRC AI Deep Dive

The NRC AI tool operates in a two-stage process:

**Stage 1 — Pattern detection (pure code, no AI)**
Pure Python/pandas grouping on the ticket DataFrame. This finds clusters by:
- `recurring_issue`: Same account + issue type appearing 2+ times
- `repeat_contact`: Same technician handling multiple tickets for the same account
- `same_day_burst`: 3+ tickets from the same account on the same calendar day

**Stage 2 — Recommendation generation (LLM)**
For each significant cluster, the system builds a structured prompt containing:
- Ticket count, recurrence rate, % of the account's total tickets
- How many unique contacts/users are affected
- A representative sample of up to 15 tickets selected for sub-issue variety and description detail
- Historical context: when this pattern first appeared, all-time count, and trend vs. the prior equivalent period

The LLM (HatzAI) reads the free-text descriptions and outputs structured recommendations (Automation, User Training, Process Change, or Sales Opportunity) aimed at preventing the ticket pattern from recurring.

### Importing Tickets for NRC AI

Currently, the tool analyzes tickets imported via CSV from Autotask.

The CSV must include these columns:
`Ticket Number, Title, Description, Account, Resources, Status, Created, Total Hours Worked, Billed Hours, Sub-Issue Type, Issue Type`

To import files into the local SQLite database for analysis:
```bash
python main.py import data/tickets.csv
```
The importer deduplicates by ticket number, so re-importing the same file is safe. Spam/phishing tickets are automatically excluded on import.
