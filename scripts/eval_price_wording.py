"""
Live eval: does the assistant say WHICH price it is quoting, and never
invent one?

The cost price (what the shop pays its supplier) and the sale price (what a
customer pays) are separate fields. The system prompt tells the model to
quote the sale price when one is set, to say plainly when only a cost price
exists, and never to state a price that isn't in the tool result. A prompt
rule can't be unit-tested -- so this asks real rate questions against a
small seeded inventory and checks the replies.

It calls the real LLM (costs a few cents, needs network and the API key in
.env), so it is NOT part of `unittest discover`. It uses a throwaway
database and never touches inventory.db, product_images/ or usage.db.

Run from the directory that holds .env (config.PROJECT_ROOT is the cwd):
    python3 /path/to/scripts/eval_price_wording.py            # one pass
    python3 /path/to/scripts/eval_price_wording.py -n 3       # 3 passes, catches flaky wording
    python3 /path/to/scripts/eval_price_wording.py -v         # also print every reply

Exit status: 1 if any reply had the wrong wording, 2 if only LLM errors
(after retries) got in the way, 0 if everything passed.
"""

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import inventory_db  # noqa: E402

SEED = [
    # name, cost, sale, stock
    ("Eval Air Filter", 380, None, 5),  # imported cost only
    ("Eval Brake Pad", None, 450, 4),  # sale price only
    ("Eval Clutch Wire", 90, 140, 12),  # both
    ("Eval Fuel Pipe", None, None, 7),  # neither
]


def mentions(reply, number):
    return re.search(rf"(?<![\d,.]){number}(?![\d]|\.\d)", reply.replace(",", "")) is not None


def labelled(reply, number, word):
    """True when the number appears within a short span of the given label word."""
    text = reply.replace(",", "")
    pattern = rf"{word}[^.\n]{{0,40}}(?<![\d.]){number}(?!\d)|(?<![\d.]){number}(?!\d)[^.\n]{{0,40}}{word}"
    return re.search(pattern, text, re.IGNORECASE) is not None


def says_sale_missing(reply):
    return re.search(
        r"(sale|selling|bikri)[^.\n]{0,40}(not set|nahi|no sale|not available|missing|hasn't|isn't|unset)"
        r"|(not set|nahi|no)[^.\n]{0,25}(sale|selling)",
        reply,
        re.IGNORECASE,
    ) is not None


def says_price_missing(reply):
    return re.search(r"not set|nahi|no price|not available|missing|unset|don't have|doesn't have", reply, re.IGNORECASE) is not None


# Each check returns an error string, or None when the reply is fine.

def both_quotes_sale(reply):
    if not mentions(reply, 140):
        return "should quote the sale price 140"
    if not re.search(r"sale|selling|bikri", reply, re.IGNORECASE):
        return "never says it is the sale price"
    if mentions(reply, 90) and not labelled(reply, 90, "cost"):
        return "mentions 90 without calling it the cost price"
    return None


def both_quotes_cost(reply):
    if not mentions(reply, 90):
        return "should quote the cost price 90"
    if not re.search(r"cost", reply, re.IGNORECASE):
        return "never says it is the cost price"
    if mentions(reply, 140) and not labelled(reply, 140, "sale|selling"):
        return "mentions 140 without calling it the sale price"
    return None


def cost_only(reply):
    if not mentions(reply, 380):
        return "should mention the cost price 380"
    if not labelled(reply, 380, "cost"):
        return "380 is not labelled as the cost price"
    # "sale price set nahi hai, cost price Rs 380" is the right answer, so only
    # a sale label directly in front of the number counts as a mislabel.
    if re.search(r"(sale|selling)\s*(price|rate)?\s*(is|of|:|=|hai)?\s*(rs\.?|₹)?\s*380", reply, re.IGNORECASE):
        return "calls the cost price 380 a sale price"
    if not says_sale_missing(reply):
        return "doesn't say the sale price is not set"
    return None


def sale_only(reply):
    if not mentions(reply, 450):
        return "should quote the sale price 450"
    if not re.search(r"sale|selling|bikri", reply, re.IGNORECASE):
        return "never says it is the sale price"
    return None


def no_price(reply):
    if re.search(r"(?:rs\.?|₹|rupees?)\s*\d", reply, re.IGNORECASE) or re.search(r"\d+\s*(?:rs|rupees?|₹)", reply, re.IGNORECASE):
        return "invents a price for an item with none"
    if not says_price_missing(reply):
        return "doesn't say the price is not set"
    return None


def unknown_item(reply):
    if re.search(r"(?:rs\.?|₹|rupees?)\s*\d", reply, re.IGNORECASE) or re.search(r"\d+\s*(?:rs|rupees?|₹)", reply, re.IGNORECASE):
        return "invents a price for an item that doesn't exist"
    return None


CASES = [
    ("both: Hinglish rate", "Eval Clutch Wire ka rate kitna hai", both_quotes_sale),
    ("both: English price", "What is the price of Eval Clutch Wire?", both_quotes_sale),
    ("both: customer price", "Eval Clutch Wire kitne mein bechna hai?", both_quotes_sale),
    ("both: asks cost", "What is the cost price of Eval Clutch Wire?", both_quotes_cost),
    ("both: asks purchase", "Eval Clutch Wire kitne ka kharida tha?", both_quotes_cost),
    ("cost only: Hinglish rate", "Eval Air Filter ka rate kitna hai", cost_only),
    ("cost only: English price", "How much does Eval Air Filter sell for?", cost_only),
    ("sale only: Hinglish rate", "Eval Brake Pad ka rate kitna hai", sale_only),
    ("neither: rate", "Eval Fuel Pipe ka rate kitna hai", no_price),
    ("unknown item: rate", "Eval Gearbox Oil ka rate kitna hai", unknown_item),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-n", "--repeat", type=int, default=1, help="passes over all cases (default 1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every reply, not just failures")
    args = parser.parse_args()

    tmp = tempfile.TemporaryDirectory(prefix="shop-eval-")
    inventory_db.INVENTORY_DB = Path(tmp.name) / "inventory.db"

    for name, cost, sale, stock in SEED:
        inventory_db.add_item(name, category="Eval", initial_quantity=stock, cost_price=cost, sale_price=sale)

    from agent.responder import Responder

    responder = Responder()
    failures = 0
    errors = 0
    total = 0

    for run in range(1, args.repeat + 1):
        for label, question, check in CASES:
            total += 1
            reply, error = "", None

            # Provider hiccups (503 "high demand") say nothing about the
            # wording, so retry them and report what's left as ERROR.
            for attempt in range(3):
                try:
                    reply = responder.respond(question, history=None)["reply"]
                    error = None
                    break
                except Exception as exc:
                    error = f"error: {' '.join(str(exc).split())[:150]}"
                    # A 429 is a per-minute quota: a couple of seconds won't clear it.
                    time.sleep((15 if "429" in error else 2) * (attempt + 1))

            if error:
                errors += 1
                problem = error
            else:
                problem = check(reply)
                if problem:
                    failures += 1

            status = "ERR " if error else ("FAIL" if problem else "ok  ")
            print(f"[{status}] run {run} | {label}")

            if problem or args.verbose:
                print(f"         Q: {question}")
                print(f"         A: {' '.join(reply.split())}")
            if problem:
                print(f"         -> {problem}")

    print(f"\n{total - failures - errors}/{total} passed, {failures} wording failures, {errors} LLM errors")
    tmp.cleanup()
    return 1 if failures else (2 if errors else 0)


if __name__ == "__main__":
    sys.exit(main())
