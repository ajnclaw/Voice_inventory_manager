class ApprovalPolicy:

    # Read-only -- safe to always run without a second thought.
    AUTO_APPROVED_TOOLS = {
        "search_items",
        "check_stock",
        "list_inventory",
        "list_low_stock",
        "sales_report",
    }

    # Changes the actual inventory records. Still routed through
    # approval_manager.py (so every write gets a printed/logged
    # "APPROVAL REQUIRED" audit line), but -- unlike a hypothetical
    # slow/irreversible action -- these ARE this agent's core, routine
    # job, so the default deployment auto-approves them rather than
    # permanently blocking the agent's main purpose whenever no
    # terminal is attached. See approval_manager.py's NEVER_AUTO_APPROVE
    # (empty here) for the knob that would change that.
    APPROVAL_REQUIRED_TOOLS = {
        "add_item",
        "rename_item",
        "record_sale",
        "record_restock",
        "adjust_stock",
    }

    def requires_approval(self, tool_name):
        return tool_name in self.APPROVAL_REQUIRED_TOOLS

    def is_allowed(self, tool_name):
        return (
            tool_name in self.AUTO_APPROVED_TOOLS
            or tool_name in self.APPROVAL_REQUIRED_TOOLS
        )

    def check(self, tool_name):
        if not self.is_allowed(tool_name):
            return {
                "allowed": False,
                "requires_approval": False,
                "reason": f"Unknown or unregistered tool: {tool_name}",
            }

        if self.requires_approval(tool_name):
            return {
                "allowed": False,
                "requires_approval": True,
                "reason": f"Tool '{tool_name}' requires approval.",
            }

        return {
            "allowed": True,
            "requires_approval": False,
            "reason": None,
        }
