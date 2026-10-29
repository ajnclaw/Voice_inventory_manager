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

Tests (auth layer):

```bash
.venv/bin/python3 -m unittest discover -s tests -v
```

## HTTP API

```
GET    /                      web UI
GET    /health                liveness
GET    /inventory             catalog + totals (no LLM call)
POST   /chat                  {"message": "..."} -> {"reply", "status"}
POST   /upload                file (CSV / PDF / photo) -> extracted candidates, no writes
POST   /import/confirm        {"items": [...]} -> applies reviewed items
POST   /import/cancel         rejection signal for monitoring
POST   /item/confirm-add      accept an agent-proposed new item
POST   /item/reject-add       reject it
POST   /inventory/adjust-stock, /inventory/set-price   inline table edits
DELETE /history               clear conversation history
GET    /usage /transcript /stats   cost, saved chats, failure rates
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
