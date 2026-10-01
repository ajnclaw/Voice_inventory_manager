import os


# Nothing in this agent's tool list is slow/resource-heavy, runs an
# arbitrary command, or publishes externally the way ~/ai-agent's
# create_video/run_command/post_to_youtube are -- every write here
# (a sale, a restock, a correction) is cheap, fast, and reversible via
# another ledger entry. So nothing is hard-blocked at this layer; whether
# writes need a real y/N answer is controlled entirely by
# AGENT_AUTO_APPROVE at the deployment level (see agent_server.py) --
# set it to leave it, unset/run with a real terminal attached to get a
# genuine per-call prompt (e.g. while testing locally before deploying).
NEVER_AUTO_APPROVE = set()


class ApprovalManager:

    def request_approval(self, tool_name, arguments, preview=None):
        print("\n" + "=" * 60)
        print("APPROVAL REQUIRED")
        print("=" * 60)

        print(f"Tool: {tool_name}")
        print("Arguments:")

        for key, value in arguments.items():
            print(f"  {key}: {value}")

        if preview:
            print()
            print(preview)

        print("=" * 60)

        if (
            os.environ.get("AGENT_AUTO_APPROVE") == "1"
            and tool_name not in NEVER_AUTO_APPROVE
        ):
            print("Auto-approved (AGENT_AUTO_APPROVE=1).")
            return True

        try:
            answer = input(
                "Allow this tool call? [y/N]: "
            ).strip().lower()
        except EOFError:
            # No interactive terminal attached (e.g. running as a headless
            # server) -- default to denying rather than hanging or crashing.
            print("No terminal attached to answer -- denying by default.")
            return False

        return answer in {"y", "yes"}
