# tools.py
#
# The entire tool surface for this agent -- inventory only. No files, no
# shell, no video, no memory-facts: those belong to a different agent
# (~/ai-agent), not this one. Keeping this list small and domain-only is
# deliberate -- see the project README for why.

from . import inventory_db
from .approval import ApprovalPolicy
from .approval_manager import ApprovalManager


def tool_result(success, output=None, error=None, denied=False):
    return {
        "success": success,
        "output": output,
        "error": error,
        "denied": denied,
    }


def _wrap(fn, **kwargs):
    try:
        output = fn(**kwargs)
        return tool_result(success=True, output=output)
    except Exception as exc:
        return tool_result(success=False, error=str(exc))


def rename_item_tool(item, new_name):
    return _wrap(inventory_db.rename_item, old_name=item, new_name=new_name)


def add_item_tool(
    name,
    category=None,
    unit="each",
    initial_quantity=0,
    cost_price=None,
    sale_price=None,
    reorder_threshold=None,
):
    return _wrap(
        inventory_db.add_item,
        name=name,
        category=category,
        unit=unit,
        initial_quantity=initial_quantity,
        cost_price=cost_price,
        sale_price=sale_price,
        reorder_threshold=reorder_threshold,
    )


def search_items_tool(query):
    return _wrap(inventory_db.search_items, query=query)


def record_sale_tool(item, quantity, unit_price=None, note=None):
    return _wrap(
        inventory_db.record_sale,
        item_name=item,
        quantity=quantity,
        unit_price=unit_price,
        note=note,
    )


def record_restock_tool(item, quantity, unit_cost=None, note=None):
    return _wrap(
        inventory_db.record_restock,
        item_name=item,
        quantity=quantity,
        unit_cost=unit_cost,
        note=note,
    )


def adjust_stock_tool(item, new_quantity, reason):
    return _wrap(
        inventory_db.adjust_stock,
        item_name=item,
        new_quantity=new_quantity,
        reason=reason,
    )


def check_stock_tool(item):
    return _wrap(inventory_db.check_stock, item_name=item)


def list_inventory_tool():
    return _wrap(inventory_db.list_inventory)


def list_low_stock_tool():
    return _wrap(inventory_db.list_low_stock)


def sales_report_tool(since):
    return _wrap(inventory_db.sales_report, since_iso=since)


TOOL_FUNCTIONS = {
    "add_item": add_item_tool,
    "rename_item": rename_item_tool,
    "search_items": search_items_tool,
    "record_sale": record_sale_tool,
    "record_restock": record_restock_tool,
    "adjust_stock": adjust_stock_tool,
    "check_stock": check_stock_tool,
    "list_inventory": list_inventory_tool,
    "list_low_stock": list_low_stock_tool,
    "sales_report": sales_report_tool,
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": (
                "Register a new catalog item. Use this only when the item "
                "genuinely doesn't exist yet -- check with search_items "
                "first. Price fields are optional; leave them out if the "
                "user doesn't want to track cost/margin for this item."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The item's name, used as its unique identifier.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional free-text category, e.g. 'tractor part'.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measure, e.g. 'each', 'kg', 'litre'. Defaults to 'each'.",
                    },
                    "initial_quantity": {
                        "type": "number",
                        "description": "Starting stock count, if known. Defaults to 0.",
                    },
                    "cost_price": {
                        "type": "number",
                        "description": "What this item costs to acquire, if the user wants margin tracking.",
                    },
                    "sale_price": {
                        "type": "number",
                        "description": "What this item sells for, if the user wants margin tracking.",
                    },
                    "reorder_threshold": {
                        "type": "number",
                        "description": "Stock level at or below which this item shows up as low-stock.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_item",
            "description": (
                "Corrects an existing item's name -- quantity, price, "
                "and its transaction history are untouched, only the "
                "label changes. Use this when an item's stored name "
                "turns out to be wrong, ambiguous, or was misread "
                "during an import (e.g. the owner says 'that part is "
                "actually called X, not Y'). Resolve the exact current "
                "name with search_items first if there's any doubt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The item's exact current stored name.",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "The corrected name to rename it to.",
                    },
                },
                "required": ["item", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_items",
            "description": (
                "Fuzzy-search the catalog by name or category. ALWAYS call "
                "this to resolve a spoken/typed item description to its "
                "exact stored name before calling record_sale, "
                "record_restock, adjust_stock, or check_stock -- never "
                "guess the exact name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Part of the item's name or category to search for.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_sale",
            "description": "Record that some quantity of an item was sold, decreasing its stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The item's exact stored name (resolve with search_items first).",
                    },
                    "quantity": {
                        "type": "number",
                        "description": "How many units were sold.",
                    },
                    "unit_price": {
                        "type": "number",
                        "description": "Price per unit for this sale, if known.",
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional free-text note about this sale.",
                    },
                },
                "required": ["item", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_restock",
            "description": "Record new stock coming in for an item, increasing its stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The item's exact stored name (resolve with search_items first).",
                    },
                    "quantity": {
                        "type": "number",
                        "description": "How many units came in.",
                    },
                    "unit_cost": {
                        "type": "number",
                        "description": "Cost per unit for this restock, if known.",
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional free-text note, e.g. supplier name.",
                    },
                },
                "required": ["item", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_stock",
            "description": (
                "Correct an item's stock count directly, e.g. after a "
                "miscount or finding damaged goods -- NOT for a normal "
                "sale or restock, those have their own tools. A reason is "
                "required every time, since this is the audit-trail tool "
                "for 'something doesn't match reality.'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The item's exact stored name (resolve with search_items first).",
                    },
                    "new_quantity": {
                        "type": "number",
                        "description": "The corrected, actual stock count.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the count is being corrected -- required.",
                    },
                },
                "required": ["item", "new_quantity", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Look up one item's current quantity and its recent transaction history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The item's exact stored name (resolve with search_items first).",
                    },
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_inventory",
            "description": "List every item in the catalog with its current quantity.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_low_stock",
            "description": "List items at or below their reorder threshold.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sales_report",
            "description": (
                "Summarize sales (quantity and, where prices are known, "
                "revenue) since a given point in time. Resolve relative "
                "language like 'this week' or 'this month' into an "
                "absolute ISO 8601 datetime yourself first, using the "
                "current time you've been given -- never pass the "
                "relative phrase through as-is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "since": {
                        "type": "string",
                        "description": "Absolute ISO 8601 datetime to report sales from.",
                    },
                },
                "required": ["since"],
            },
        },
    },
]


class ToolManager:
    def __init__(self):
        self.approval_policy = ApprovalPolicy()
        self.approval_manager = ApprovalManager()
        self.functions = TOOL_FUNCTIONS
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def execute(self, tool_name, arguments):
        result = self._execute(tool_name, arguments)

        if self.logger:
            self.logger.log_tool_call(tool_name, arguments, result)

        return result

    def _execute(self, tool_name, arguments):
        policy = self.approval_policy.check(tool_name)

        if not policy["allowed"] and not policy["requires_approval"]:
            return tool_result(success=False, error=policy["reason"])

        if policy["requires_approval"]:
            approved = self.approval_manager.request_approval(tool_name, arguments)

            if not approved:
                return tool_result(
                    success=False,
                    error=f"Tool '{tool_name}' was denied by the user.",
                    denied=True,
                )

        function = self.functions.get(tool_name)

        if function is None:
            return tool_result(success=False, error=f"Tool not found: {tool_name}")

        try:
            result = function(**arguments)
            return (
                result
                if isinstance(result, dict)
                else tool_result(success=True, output=result)
            )
        except Exception as exc:
            return tool_result(success=False, error=str(exc))
