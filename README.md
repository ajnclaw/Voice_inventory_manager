# Voice Inventory Manager

A chat/voice inventory assistant for a small farm-equipment spare-parts
shop. The owner asks in plain language ("how many oil filters do I
have?", "sold 3 of the 12mm bolts", "what's running low?") instead of
maintaining a spreadsheet, and bulk-loads stock by uploading a supplier
PDF, a CSV, or a photo of a handwritten list.

Built for a real shop and running in production, so the design leans
toward **exact answers, an auditable ledger, and a human in the loop for
anything that changes the catalog**.

<!-- TODO: demo GIF / screenshots go here -->

## What it does

- **Chat with your stock.** Search, check quantity, record sales and
  restocks, correct mistakes, list low-stock items, get a sales report.
- **Bulk import with review.** Upload a PDF catalog, CSV, or photo. The
  LLM extracts candidate items (including product pictures from PDFs),
  flags ones that already exist, and shows an editable preview. Nothing
  is written until the owner confirms.
- **Owner approval for new items.** If the chat agent wants to add an
  item, it is intercepted and shown as an Accept/Reject card; the agent
  can never create catalog entries on its own.
- **Table view.** Filter, sort, and edit stock/price inline, no LLM call.
- **Observability.** Per-request traces, token/cost tracking, a saved
  transcript, and failure/reject-rate stats (`/usage`, `/transcript`, `/stats`).

## Architecture

```
 Browser (single-page UI, web/index.html)
        |  HTTPS (Caddy + Let's Encrypt)
        v
 agent_server.py  (stdlib http.server, Google sign-in + allowlist)
        |
        v
 agent/responder.py  -- one tool-calling loop, small tool list
        |                         |
        v                         v
 agent/tools.py             agent/llm_client.py -> any OpenAI-compatible API
        |                         (Gemini, OpenRouter, DeepSeek, Ollama, ...)
        v
 agent/inventory_db.py  (SQLite: items cache + append-only transactions ledger)
```

Design decisions and trade-offs:

- **Structured SQLite, not vector memory.** "How many filters do I
  have" needs an exact number, not a plausible nearest neighbour.
  `items` is a cached view of current stock; `transactions` is the
  append-only source of truth for every sale, restock, and correction,
  so nothing is silently overwritten and everything is auditable.
- **One responder, no planner/executor pipeline.** Every request is a
  single resolve-then-act step over a small tool list, so the
  tool-calling loop alone covers it. Less machinery means fewer places
  for the model to misroute.
- **Small, domain-only tools:** `add_item`, `rename_item`, `set_category`,
  `set_price`, `search_items`, `record_sale`, `record_restock`,
  `adjust_stock`, `check_stock`, `list_inventory`, `list_low_stock`,
  `sales_report`. No file, shell, or network tools.
- **Routine writes are auto-approved; catalog changes are not.** A wrong
  sale or restock is cheap to fix with a corrective ledger entry
  (`AGENT_AUTO_APPROVE`, see `agent/approval_manager.py`). Adding items
  and bulk imports always need explicit owner review.
- **Minimal dependencies.** Python stdlib plus PyMuPDF for PDFs. Image
  and handwriting reading goes through the same hosted multimodal LLM as
  chat, so there is no local model or GPU and it runs on a t3.micro.
- **Auth fails closed.** Google sign-in plus a mandatory email allowlist,
  re-checked on every request, HMAC-signed session cookie, CSRF header on
  cookie-authenticated writes. The server will not accept requests
  unless `GOOGLE_CLIENT_ID`, `ALLOWED_GOOGLE_EMAILS` and `SESSION_SECRET`
  are all set. A bearer token remains for scripts.

## Running it

```bash
cp .env.example .env     # fill in an LLM key, auth settings
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python3 agent_server.py     # 0.0.0.0:8766 by default; set AGENT_HOST=127.0.0.1 behind a proxy
```

Run it from the project root: `inventory.db`, `usage.db`,
`monitoring.db`, `transcripts.jsonl`, `logs/` and `product_images/` are
created in the current directory. See [DATA.md](DATA.md) for what lives
where. Back up `inventory.db`; it is the shop's real records.

Tests (auth, photo/prices, admin dashboard). They use throwaway databases
and never touch the real data files:

```bash
.venv/bin/python3 -m unittest discover -s tests -v
```

## Admin dashboard

`/admin` is a separate page, visible only to admins, that shows how well
the assistant is doing and where it falls short. Sections are ordered by
importance: quality and gaps, cost, usage, speed, feature adoption, data
health, security. Click a user for their totals and every session, turn by
turn (message, reply, status, tokens, cost, latency, tools called), with
problem turns highlighted and a step-by-step trace for any run.

**Access.** Set `ADMIN_GOOGLE_EMAILS` (comma-separated). Every admin must
also be on `ALLOWED_GOOGLE_EMAILS`; an address listed only as admin is
ignored. Unset or empty means nobody is an admin. The shared bearer token is
**never** an admin, and with Google sign-in off there are no admins. Signed
out gets 401, signed in but not an admin gets 403 (both recorded as auth
failures). The dashboard routes are read-only `GET`s; the one new write,
`POST /feedback`, follows the same CSRF-header rule as every other
cookie-authenticated write. The admin button in the chat header appears only
for admins.

**Who did what.** Chat turns, LLM calls, tool calls, imports, table edits,
photo uploads, feedback and sign-ins now carry the signed-in email (the
bearer token is recorded as its own identity, `bearer-token`) and a session
id. A session is one user's activity with no gap over `SESSION_IDLE_MINUTES`
(default 30). Conversation history is now **per user** instead of one list
shared by everyone, and `DELETE /history` clears only the caller's own.
`/usage`, `/transcript` and `/stats` show every user's activity, so they are
now limited to admins and the bearer token.

**Existing data.** The new columns and tables are added to the existing
SQLite files in place, additive only, on first start; nothing is rewritten
or dropped, and `inventory.db` is not touched. Rows from before this change
have no user and show as "(unattributed)"; their sessions are inferred from
gaps in the timeline and marked "inferred".

**Reading the numbers.**

- Cost comes from the price table in `agent/pricing.py`. A model that is not
  in it has token counts but no cost; the page says how many calls are
  unpriced, so the dollar total is then a lower bound.
- "Couldn't find / do", "asked which one" and "corrected by the user" are
  keyword and timing heuristics over the replies (English and Hinglish),
  defined at the top of `agent/admin_metrics.py`. They point at turns worth
  reading, not exact counts. Thumbs up/down appear under each reply in the
  chat.
- Latency is server-side only (LLM + tool time); browser speech-to-text and
  playback are not included, and turns from before timing was recorded have
  none.
- Days and hours use `ADMIN_TIMEZONE` (default `Asia/Kolkata`).

## HTTP API

```
GET    /                      web UI
GET    /health                liveness
GET    /inventory             catalog + totals (no LLM call)
POST   /chat                  {"message": "..."} -> {"reply", "status", "run_id"}
POST   /feedback              {"run_id", "rating": "up"|"down"} thumbs on a reply
POST   /upload                file (CSV / PDF / photo) -> extracted candidates, no writes
POST   /import/confirm        {"items": [...]} -> applies reviewed items
POST   /import/cancel         rejection signal for monitoring
POST   /item/confirm-add      accept an agent-proposed new item
POST   /item/reject-add       reject it
POST   /inventory/adjust-stock, /inventory/set-price   inline table edits
DELETE /history               clear conversation history
GET    /usage /transcript /stats   cost, saved chats, failure rates (admins / bearer token)
GET    /admin                 admin dashboard page (admins only)
GET    /admin/api/overview?days=30      dashboard data (0 = all time)
GET    /admin/api/user?email=...&days=  one user's totals and sessions
GET    /admin/api/turn?run_id=...       step-by-step trace of one run
```

## Deployment

Runs as a systemd service behind Caddy (automatic HTTPS) on a small
always-on VPS, bound to `127.0.0.1`, with a per-minute health check that
restarts the service after repeated failures.

## What's next

- Seeded demo mode with fake inventory, so the app can be shown without real data
- Evaluation set (labelled queries, sample PDFs) with accuracy and cost numbers
- More tests (ledger, add-item interception, import grouping) and CI
- Nightly off-box backups of `inventory.db` and `product_images/`
- Optional Postgres backend
