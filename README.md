# nikki_hansi_shop

A voice/chat inventory assistant for a farm equipment shop. The owner
talks to it (search the catalog, record a sale, check what's low on
stock, get a sales report) instead of managing a spreadsheet he doesn't
currently keep at all.

## Architecture

This is a separate, independent project from `~/ai-agent` (the
personal assistant it was forked out of) -- different user, different
data, different risk profile, and deliberately a different tool list.
See the top-level discussion in that project's conversation history
for the full reasoning; the short version:

- **Structured SQLite, not vector memory.** "How many filters do I
  have" needs an exact answer, not a nearest-neighbor-plausible one.
  `agent/inventory_db.py` has two tables: `items` (a cached view of
  current stock) and `transactions` (the real source of truth -- an
  append-only ledger of every sale, restock, and correction, so
  nothing is ever silently overwritten and everything is auditable).

- **One conversational responder, no planner/executor pipeline.**
  `~/ai-agent` has a plan -> execute -> verify -> recover pipeline for
  multi-step tasks. Every real bug that came up while building that
  project lived in that pipeline (a planner misreading which tool a
  request needed). This project's tool list is small and every request
  is a single resolve-then-act step, so `agent/responder.py`'s own
  tool-calling loop (which already supports calling several tools in
  sequence within one turn) covers everything a planner would, without
  taking on that extra failure surface. `agent_server.py` just calls
  `Responder.respond()` directly.

- **Small, domain-only tool list** (`agent/tools.py`): `add_item`,
  `rename_item`, `search_items`, `record_sale`, `record_restock`,
  `adjust_stock`, `check_stock`, `list_inventory`, `list_low_stock`,
  `sales_report`. Nothing else -- no file tools, no shell, no video, no
  memory-facts. A small, non-overlapping tool list is also just easier
  for the LLM to route correctly; that's not only a safety argument.

- **Minimal dependencies.** Started stdlib-only; PyMuPDF was added for
  bulk import (PDF text extraction / page rendering). Still no local
  model, no GPU -- image and handwriting reading goes through the same
  hosted multimodal LLM already used for chat, not a separate OCR
  engine. That's what keeps this cheap to host anywhere.

- **Approval is auto-approved for inventory writes by design**, not
  because writes don't matter, but because recording a sale/restock/
  correction is this agent's routine job -- a mistake is cheap to
  correct with another ledger entry (`adjust_stock`), so blocking on a
  confirmation that can't realistically be answered (no terminal on a
  phone-facing server) would just make the core feature unusable. See
  `agent/approval_manager.py`. The knob is `AGENT_AUTO_APPROVE` --
  unset it (or run with a real terminal attached) to get a genuine
  per-call y/N prompt instead, e.g. while testing locally.

## Running it

```bash
cp .env.example .env              # fill in OPENROUTER_API and AGENT_API_TOKEN
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python3 agent_server.py  # listens on 0.0.0.0:8766
```

Every request needs `Authorization: Bearer <AGENT_API_TOKEN>` -- the
server refuses to even start without that token configured. This
isn't a per-user login system, just one shared secret between the
server and the one phone app that talks to it, so a public EC2/ECS
endpoint isn't wide open to anyone who finds the URL. See
`agent_server.py`'s `_authorized()`.

```
POST /chat             {"message": "..."} -> {"reply": "...", "status": "..."}
POST /upload           {"filename", "mime_type", "content_base64", "hint"?} -> extracted items (no writes)
POST /import/confirm   {"items": [...]} -> applies them, returns per-item results
POST /import/cancel    {"item_count"} -> backend-only rejection signal, nothing shown in the app
DELETE /history         -> clears conversation history
GET /health              -> {"status": "ok", "turns": N}
GET /inventory           -> full catalog + summary totals (table view, no LLM call)
GET /usage               -> cumulative LLM tokens/cost, all time
GET /transcript          -> saved conversation log
GET /stats               -> backend-only: auth failures, failure rate, import reject rate
```

`/upload` accepts a text/CSV file, a PDF, or a photo and extracts
candidate sales/restocks/corrections/new items WITHOUT writing
anything -- the web UI shows a preview, and only `/import/confirm`
(after the owner reviews it) actually touches `inventory.db`. `hint`
("buy", "sale", or "catalog") tells the extractor upfront what kind of
document this is, overriding its own judgment -- see
`agent/import_extractor.py`.

`inventory.db` and `logs/` (plus `usage.db`, `monitoring.db`,
`transcripts.jsonl`) are created in the current working directory on
first run -- always run this from the project root. See
[DATA.md](DATA.md) for exactly what's stored where.

## Hosting

Not meant to run behind `adb reverse` the way the personal Jarvis
project does (that only works because *that* phone is physically
tethered to *that* PC). This needs a real network address the owner's
phone can reach directly -- a small always-on VPS with this process
run as a systemd service, behind Caddy or similar for automatic HTTPS,
is the intended deployment. Back up `inventory.db` regularly (it's the
shop's actual business records) -- a simple cron copy or something
like Litestream.
