# db_migrate.py
#
# Additive-only schema upgrades for the SQLite files that already hold
# real data. CREATE TABLE IF NOT EXISTS never touches an existing table,
# so a column added later has to be appended explicitly -- and only ever
# appended: nothing here renames, drops or rewrites a column or a row.
# Rows written before a column existed simply read NULL there, which the
# admin dashboard shows as "unattributed".


def ensure_columns(conn, table, columns):
    """
    columns: {name: "TYPE [DEFAULT ...]"}. Adds whichever are missing,
    leaves existing ones (and all rows) alone. Safe to call on every
    connect: PRAGMA table_info is the check, so a repeat call is a no-op
    rather than an error to swallow.
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    for name, declaration in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
