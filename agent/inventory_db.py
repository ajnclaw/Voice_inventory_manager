# inventory_db.py
#
# Inventory lives in a small SQLite database (nikki_hansi_shop/inventory.db),
# not vector memory -- "how many filters do I have" needs an exact,
# deterministic answer, not a nearest-neighbor-plausible one.
#
# `transactions` is the real source of truth: an append-only ledger of
# every sale, restock, and correction, each with its own price snapshot
# and timestamp -- nothing is ever deleted or overwritten here, so the
# full history is always reconstructable and auditable. `items` is a
# cached, materialized view (current_quantity is always just the sum of
# that item's transactions.quantity_delta) kept in sync on every insert
# purely as a read optimization, never as the authority.

import sqlite3
from datetime import datetime, timezone

from .config import INVENTORY_DB


class ItemNotFound(ValueError):
    pass


class DuplicateItem(ValueError):
    pass


def _connect():
    conn = sqlite3.connect(INVENTORY_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            category TEXT,
            unit TEXT NOT NULL DEFAULT 'each',
            current_quantity REAL NOT NULL DEFAULT 0,
            reorder_threshold REAL,
            cost_price REAL,
            sale_price REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL REFERENCES items(id),
            type TEXT NOT NULL CHECK(type IN ('sale', 'restock', 'adjustment')),
            quantity_delta REAL NOT NULL,
            unit_price REAL,
            note TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat()


def _item_row_to_dict(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "category": row["category"],
        "unit": row["unit"],
        "current_quantity": row["current_quantity"],
        "reorder_threshold": row["reorder_threshold"],
        "cost_price": row["cost_price"],
        "sale_price": row["sale_price"],
    }


def _get_item_row(conn, item_name):
    row = conn.execute(
        "SELECT * FROM items WHERE name = ?", (item_name,)
    ).fetchone()

    if row is None:
        raise ItemNotFound(
            f"No item named '{item_name}'. Use search_items to find the "
            f"right name, or add_item if it genuinely doesn't exist yet."
        )

    return row


def rename_item(old_name, new_name):
    """
    Corrects an item's name without touching its quantity, price, or
    transaction history -- a real recurring need, not a one-off: a
    supplier catalog import can produce a name that only turns out to
    be ambiguous once the owner actually knows the part (e.g. two
    completely different parts a sloppy price list both called
    "FLYWHEEL FAN", one of which was actually a flywheel MAGNET).
    `items.name` is what transactions join against by id, not by name,
    so this is purely a label change -- the ledger stays intact.
    """
    if not new_name or not new_name.strip():
        raise ValueError("New name can't be empty.")

    new_name = new_name.strip()

    with _connect() as conn:
        _get_item_row(conn, old_name)  # raises ItemNotFound if missing

        clash = conn.execute(
            "SELECT id FROM items WHERE name = ?", (new_name,)
        ).fetchone()

        if clash:
            raise DuplicateItem(
                f"An item named '{new_name}' already exists -- pick a "
                f"different name."
            )

        conn.execute(
            "UPDATE items SET name = ? WHERE name = ?", (new_name, old_name)
        )

    return {"old_name": old_name, "new_name": new_name}


def set_category(item_name, category):
    """
    Corrects an item's category -- e.g. a bulk-import extraction that
    initially missed or mislabeled the product-family heading a part
    actually belongs to (see agent/import_extractor.py's category
    rule). Quantity, price, and transaction history are untouched.
    """
    with _connect() as conn:
        _get_item_row(conn, item_name)  # raises ItemNotFound if missing

        conn.execute(
            "UPDATE items SET category = ? WHERE name = ?",
            (category, item_name),
        )

    return {"item": item_name, "category": category}


def add_item(
    name,
    category=None,
    unit="each",
    initial_quantity=0,
    cost_price=None,
    sale_price=None,
    reorder_threshold=None,
):
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM items WHERE name = ?", (name,)
        ).fetchone()

        if existing:
            raise DuplicateItem(
                f"An item named '{name}' already exists -- use "
                f"record_restock/adjust_stock to change its quantity, "
                f"not add_item again."
            )

        created_at = _now()

        cursor = conn.execute(
            "INSERT INTO items "
            "(name, category, unit, current_quantity, reorder_threshold, "
            " cost_price, sale_price, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                name, category, unit, initial_quantity, reorder_threshold,
                cost_price, sale_price, created_at,
            ),
        )

        item_id = cursor.lastrowid

        if initial_quantity:
            conn.execute(
                "INSERT INTO transactions "
                "(item_id, type, quantity_delta, unit_price, note, created_at) "
                "VALUES (?, 'restock', ?, ?, ?, ?)",
                (item_id, initial_quantity, cost_price, "Initial stock", created_at),
            )

    return {
        "id": item_id,
        "name": name,
        "category": category,
        "unit": unit,
        "current_quantity": initial_quantity,
        "reorder_threshold": reorder_threshold,
        "cost_price": cost_price,
        "sale_price": sale_price,
    }


def search_items(query):
    """
    Matches in both directions: the stored name containing the query
    (the original behavior -- query "filter" finds "oil filter") AND
    the query containing the stored name (query "oil filters" finds
    "oil filter", since "oil filter" is a substring of "oil filters").
    Without the second direction, a plain plural/singular mismatch --
    exactly the kind of thing a bulk-import extraction or a slightly
    different spoken phrasing produces constantly -- fails to match
    anything at all even though the item obviously exists.
    """
    query_lower = query.lower()
    query_like = f"%{query_lower}%"

    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM items WHERE lower(name) LIKE ? OR lower(category) LIKE ? "
            "OR ? LIKE '%' || lower(name) || '%' "
            "ORDER BY name",
            (query_like, query_like, query_lower),
        ).fetchall()

    return [_item_row_to_dict(row) for row in rows]


def _apply_transaction(conn, item_row, delta, txn_type, unit_price, note, created_at):
    new_quantity = item_row["current_quantity"] + delta

    conn.execute(
        "INSERT INTO transactions "
        "(item_id, type, quantity_delta, unit_price, note, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (item_row["id"], txn_type, delta, unit_price, note, created_at),
    )

    conn.execute(
        "UPDATE items SET current_quantity = ? WHERE id = ?",
        (new_quantity, item_row["id"]),
    )

    return new_quantity


def record_sale(item_name, quantity, unit_price=None, note=None):
    if quantity <= 0:
        raise ValueError("Sale quantity must be positive.")

    with _connect() as conn:
        item_row = _get_item_row(conn, item_name)

        if item_row["current_quantity"] < quantity:
            raise ValueError(
                f"Only {item_row['current_quantity']} {item_row['unit']} of "
                f"'{item_name}' in stock -- can't record a sale of {quantity}. "
                f"If the stock count itself is wrong, use adjust_stock instead."
            )

        new_quantity = _apply_transaction(
            conn, item_row, -quantity, "sale", unit_price, note, _now(),
        )

    return {
        "item": item_name,
        "sold": quantity,
        "unit_price": unit_price,
        "new_quantity": new_quantity,
    }


def record_restock(item_name, quantity, unit_cost=None, note=None):
    if quantity <= 0:
        raise ValueError("Restock quantity must be positive.")

    with _connect() as conn:
        item_row = _get_item_row(conn, item_name)

        new_quantity = _apply_transaction(
            conn, item_row, quantity, "restock", unit_cost, note, _now(),
        )

    return {
        "item": item_name,
        "restocked": quantity,
        "unit_cost": unit_cost,
        "new_quantity": new_quantity,
    }


def adjust_stock(item_name, new_quantity, reason):
    if not reason or not reason.strip():
        raise ValueError(
            "adjust_stock requires a reason -- this is the correction tool, "
            "so it needs an audit trail (e.g. 'found 3 damaged', 'miscount')."
        )

    with _connect() as conn:
        item_row = _get_item_row(conn, item_name)
        previous_quantity = item_row["current_quantity"]
        delta = new_quantity - previous_quantity

        _apply_transaction(
            conn, item_row, delta, "adjustment", None, reason, _now(),
        )

    return {
        "item": item_name,
        "previous_quantity": previous_quantity,
        "new_quantity": new_quantity,
        "reason": reason,
    }


def check_stock(item_name, recent_transactions=5):
    with _connect() as conn:
        item_row = _get_item_row(conn, item_name)

        rows = conn.execute(
            "SELECT * FROM transactions WHERE item_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (item_row["id"], recent_transactions),
        ).fetchall()

    return {
        **_item_row_to_dict(item_row),
        "recent_transactions": [
            {
                "type": row["type"],
                "quantity_delta": row["quantity_delta"],
                "unit_price": row["unit_price"],
                "note": row["note"],
                "created_at": row["created_at"],
            }
            for row in rows
        ],
    }


def list_inventory():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM items ORDER BY category, name").fetchall()

    return [_item_row_to_dict(row) for row in rows]


def inventory_overview():
    """
    The full catalog plus roll-up totals, for a dedicated human-
    readable view (a real table, not a chat reply the owner has to
    read as prose). Value is computed from sale_price where set,
    falling back to cost_price, same degrade-gracefully-when-price-
    unknown pattern as sales_report -- items with neither don't
    contribute to the total, and the total is null (not zero) if
    nothing in the whole catalog has a price on file.
    """
    items = list_inventory()

    total_units = 0.0
    total_value = 0.0
    value_known = False
    low_stock_count = 0

    for item in items:
        total_units += item["current_quantity"]

        price = (
            item["sale_price"]
            if item["sale_price"] is not None
            else item["cost_price"]
        )

        if price is not None:
            total_value += item["current_quantity"] * price
            value_known = True

        if (
            item["reorder_threshold"] is not None
            and item["current_quantity"] <= item["reorder_threshold"]
        ):
            low_stock_count += 1

    return {
        "items": items,
        "summary": {
            "total_items": len(items),
            "total_units": total_units,
            "total_value": total_value if value_known else None,
            "low_stock_count": low_stock_count,
        },
    }


def list_low_stock():
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM items WHERE reorder_threshold IS NOT NULL "
            "AND current_quantity <= reorder_threshold ORDER BY name"
        ).fetchall()

    return [_item_row_to_dict(row) for row in rows]


def sales_report(since_iso):
    """
    since_iso: an absolute ISO 8601 datetime string -- resolving relative
    language ("this week", "this month") against the current time happens
    in the system prompt/tool-calling layer, the same division of
    responsibility as ~/ai-agent's reminders.py.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT t.*, i.name as item_name FROM transactions t "
            "JOIN items i ON i.id = t.item_id "
            "WHERE t.type = 'sale' AND t.created_at >= ? "
            "ORDER BY t.created_at",
            (since_iso,),
        ).fetchall()

    total_quantity = 0
    total_revenue = 0.0
    revenue_known = False
    by_item = {}

    for row in rows:
        qty = -row["quantity_delta"]  # stored negative for sales
        total_quantity += qty

        entry = by_item.setdefault(
            row["item_name"], {"quantity": 0, "revenue": 0.0}
        )
        entry["quantity"] += qty

        if row["unit_price"] is not None:
            revenue = qty * row["unit_price"]
            total_revenue += revenue
            entry["revenue"] += revenue
            revenue_known = True

    return {
        "since": since_iso,
        "total_quantity_sold": total_quantity,
        "total_revenue": total_revenue if revenue_known else None,
        "by_item": by_item,
    }
