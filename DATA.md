# Data and logging reference

This project writes to four SQLite databases and two file-based logs,
each created fresh (empty, schema auto-created) the first time its
owning module runs. None of them are committed to git -- see
`.gitignore`. All of them live in the project root.

| File | Format | Purpose |
|---|---|---|
| `inventory.db` | SQLite | The actual business data -- what's in stock. |
| `usage.db` | SQLite | Cumulative LLM token/cost tracking, all time. |
| `monitoring.db` | SQLite | Backend-only system health (auth failures, request failure rate, import accept/reject rate). |
| `transcripts.jsonl` | JSON Lines | A continuous, human-readable conversation log. |
| `logs/<run_id>.json` | JSON (one file per request) | The detailed technical trace of one request -- full prompts, tool calls, tokens, cost. |

---

## `inventory.db` -- `agent/inventory_db.py`

The only one of the four that matters if it's lost -- this is the
shop's real business records. The other three are observability data,
regenerable from nothing; this one isn't.

### `items`
A cached, materialized view of current stock. Always derivable by
summing that item's rows in `transactions` -- never the authority on
its own.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `name` | TEXT, UNIQUE | The catalog's unique identifier for the item. |
| `category` | TEXT | Free text, optional. |
| `unit` | TEXT | `each`, `kg`, `litre`, etc. Defaults to `each`. |
| `current_quantity` | REAL | Cached running total -- see `transactions`. |
| `reorder_threshold` | REAL | Null = no low-stock alert for this item. |
| `cost_price` | REAL | What the shop pays, nullable. |
| `sale_price` | REAL | What the shop charges, nullable. |
| `created_at` | TEXT | ISO 8601. |

### `transactions`
The real source of truth: an append-only ledger. Nothing is ever
updated or deleted here -- a correction is a new row, not an edit to
an old one, so the full history is always reconstructable.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `item_id` | INTEGER, FK -> `items.id` | |
| `type` | TEXT | `sale` \| `restock` \| `adjustment` |
| `quantity_delta` | REAL | Negative for a sale, positive for a restock. For an adjustment, the signed change applied (not an absolute total). |
| `unit_price` | REAL | Snapshotted at the time of this transaction -- a later price change on `items` never rewrites history. |
| `note` | TEXT | Free text (e.g. "Bulk import", a restock's supplier, an adjustment's reason). |
| `created_at` | TEXT | ISO 8601. |

---

## `usage.db` -- `agent/usage_tracker.py`

Every LLM call this project ever makes, across every restart. Read
back via `GET /usage`.

### `llm_calls`
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `run_id` | TEXT | Matches a `logs/<run_id>.json` trace file, so you can cross-reference a specific call back to its full request/response. |
| `component` | TEXT | Which part of the code made the call -- `responder`, `import_extractor`, or `compactor`. |
| `model` | TEXT | e.g. `openai/gpt-4o-mini`. |
| `prompt_tokens` | INTEGER | |
| `completion_tokens` | INTEGER | |
| `cost_usd` | REAL | Estimated from `agent/pricing.py`'s local pricing table -- null if the model isn't in that table. Not pulled live from the provider; a running estimate, not a substitute for checking the real bill. |
| `created_at` | TEXT | ISO 8601. |

---

## `monitoring.db` -- `agent/monitoring.py`

Backend-only system health. Never surfaced in `web/index.html` --
read back via `GET /stats`, which the app's own UI never calls.

### `auth_failures`
Every rejected `Authorization` token.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `path` | TEXT | Which endpoint was hit. |
| `remote_addr` | TEXT | The caller's IP. |
| `created_at` | TEXT | ISO 8601. |

### `request_outcomes`
Completed/failed counts per endpoint, for an aggregate failure rate
you'd otherwise have to notice by manually scanning `logs/`.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `endpoint` | TEXT | `chat` or `upload`. |
| `status` | TEXT | `completed` or `failed`. |
| `created_at` | TEXT | ISO 8601. |

### `import_outcomes`
Whether a bulk-import preview actually got saved or rejected -- the
direct quality signal for the extraction feature. `cancelled` is
logged by a fire-and-forget beacon when the owner taps Cancel in the
web UI; nothing is ever shown back to them either way.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `outcome` | TEXT | `confirmed` or `cancelled`. |
| `item_count` | INTEGER | How many items were in that preview. |
| `created_at` | TEXT | ISO 8601. |

---

## `transcripts.jsonl` -- `agent/transcript_log.py`

Not a database -- plain JSON Lines (one JSON object per line), so
appends never require reading/rewriting the whole file. One entry per
`/chat` turn or `/upload` call, in order. This is "show me the whole
conversation" -- the detailed per-request traces in `logs/` are built
for debugging one request, not for reading back a conversation.

Each line:
```json
{
  "timestamp": "...",
  "user_input": "...",
  "reply": "...",
  "status": "completed",
  "model": "openai/gpt-4o-mini",
  "prompt_tokens": 1455,
  "completion_tokens": 11,
  "cost_usd": 0.00022485
}
```
`prompt_tokens`/`completion_tokens`/`cost_usd` are the *turn's* summed
totals -- a turn can involve more than one LLM call (a tool-calling
loop), not a single call's numbers.

---

## `logs/<run_id>.json` -- `agent/logger.py`

One file per request (`/chat`, `/upload`, or `/import/confirm`), named
by a timestamped run id. The detailed technical trace: every LLM call
within that request, with the **full request messages sent** (the
complete system+user prompt, not just the reply -- vision calls have
their base64 image data stripped to a placeholder so a photo upload
doesn't bloat the file), every tool call and its result, and per-call
token/cost numbers. This is what you'd open to answer "what exactly
did the model see, and why did it do that" for one specific request.
