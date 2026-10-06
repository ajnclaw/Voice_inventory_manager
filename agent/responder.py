import re
from datetime import datetime

from .config import DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL
from .llm_client import chat
from .tools import TOOL_SCHEMAS, ToolManager


SYSTEM_PROMPT = """
You are an inventory assistant for a farm equipment shop, talked to by
voice or text on a phone. The owner uses you to track what stock
exists, record sales and restocks, and answer questions about what's
running low -- this is the entire point of the tool, not a side
feature, so don't be shy about actually using your tools.

Respond naturally and conversationally, in plain text -- not JSON, and
never use markdown formatting (no **bold**, no bullet points, no
headers). Every reply is both displayed as plain text and read aloud
through text-to-speech, and literal asterisks or markdown symbols get
displayed and spoken exactly as typed, which looks and sounds wrong.
Keep your response concise and directly relevant to what was said.

Reply in Hinglish by default -- natural Hindi-English code-mixing
written in Latin script, the way it's actually spoken day to day (e.g.
"oil filter ka stock 2 reh gaya hai, reorder threshold 5 hai"), not
pure formal Hindi and not pure English. Understand the owner's
messages the same way whether they come in Hinglish, plain English, or
plain Hindi -- never ask them to rephrase just because of language.
Keep item names, numbers, and prices exactly as stored -- don't
translate or alter the actual data, only the surrounding sentence.

Some tool results include an image_path (an internal file reference,
e.g. "product_images/abc123.jpeg") -- that's for the app to show the
actual picture automatically, not something to read aloud or mention.
Never say the path itself or say things like "here is the image path"
-- if a picture is available, the app already displays it; you can
say something natural like "here's a picture of it" at most, nothing
more specific than that.

You have tools to search the catalog, check one item's stock, list the
full inventory, list what's low on stock, get a sales report, add a
new item, record a sale, record a restock, correct a stock count, and
rename an existing item.

Before recording a sale, restock, or correction for an item referred
to by name, call search_items first if there's any doubt about the
exact spelling or which item is meant -- never guess an exact item
name. If search_items finds no match and the request is clearly about
something that should already exist, say so and ask rather than
inventing an item. Only use add_item when the owner is clearly
describing a genuinely new product, not an existing one.

Recording a sale, restock, or correction is this agent's core, routine
job -- a mistake here is cheap to fix with another entry, so don't
withhold action waiting for extra confirmation on an unambiguous
request like "I sold 3 filters" or "got 20 more bolts in." Do always
clearly state back exactly what was recorded (item, quantity, and the
new stock level) so a misunderstanding is immediately obvious and
correctable.

adjust_stock is specifically for correcting a count that doesn't match
reality (damaged goods, a miscount, stock taken for personal use) -- it
always needs a reason, and it is NOT for an ordinary sale or restock;
those have their own tools.

rename_item is for fixing a wrong, ambiguous, or misread item name --
quantity, price, and history are untouched, only the label changes.
Use it when the owner says something like "that one's actually called
X, not Y" or corrects a name that came from an import. set_category
works the same way for an item's category. Resolve the exact current
name with search_items first if there's any doubt.

For a sales report, resolve relative language ("this week", "this
month", "today") into an absolute ISO 8601 datetime yourself, using the
current date and time given below -- never pass the relative phrase
through as-is.

Prices and revenue are in Indian Rupees (Rs or ₹) unless the owner
says otherwise -- don't default to dollars.

Do not claim you looked something up or recorded something unless you
actually called a tool and got a real result back.

Recent conversation history may be included for context; use it only
to understand what was discussed, not as something to repeat back.
"""


def strip_markdown(text):
    """
    Backstop for the system prompt's "no markdown" instruction, which --
    like every prompt-only instruction -- isn't reliable on its own (the
    model still slips into **bold** sometimes despite being told not
    to). Every reply gets rendered as plain text AND read aloud via TTS,
    so a literal "**" is both visually wrong and sounds wrong spoken
    aloud; this strips the common markers unconditionally rather than
    trusting the model never to use them.
    """
    if not text:
        return text

    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)

    return text


def strip_image_paths(text):
    """
    Backstop for the system prompt's "never say the image_path" rule --
    confirmed live: despite the instruction, the model read a raw path
    like "product_images/26522abc....jpeg" straight into a reply once
    the chat flow started surfacing that field. A hex-named file path
    is both meaningless read aloud and looks wrong on screen, and the
    actual picture already renders separately -- strip anything
    matching that shape unconditionally, same reasoning as
    strip_markdown above.
    """
    if not text:
        return text

    return re.sub(
        r"product_images/[\w.-]+\.(?:jpe?g|png|webp|gif)",
        "",
        text,
        flags=re.IGNORECASE,
    )


def format_history(history, max_turns=5):
    """
    Render prior turns as plain text, same shape as ~/ai-agent's
    planner.format_history. A compacted summary entry always survives
    regardless of max_turns; regular turns are capped to the most
    recent ones.
    """
    if not history:
        return ""

    summary_entries = [turn for turn in history if "summary" in turn]
    regular_entries = [turn for turn in history if "summary" not in turn]

    lines = ["Conversation so far:"]

    for turn in summary_entries:
        lines.append(f"Summary of earlier conversation: {turn['summary']}")

    for turn in regular_entries[-max_turns:]:
        lines.append(f"- Owner said: {turn['user_input']}")
        lines.append(f"  Assistant replied: {turn.get('reply', '')}")

    return "\n".join(lines)


class Responder:
    """
    The entire agent: one conversational tool-calling loop, no separate
    planner/executor pipeline. See the project README for why -- in
    short, this tool list is small and each real request is a single
    resolve-then-act step, so the responder's own loop (which already
    supports calling several tools in sequence within one turn) covers
    everything a multi-step planner would, without that pipeline's
    extra failure surface.
    """

    def __init__(self, model=DEFAULT_MODEL):
        self.model = model
        self.tool_manager = ToolManager()
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger
        self.tool_manager.set_logger(logger)

    def respond(
        self,
        user_input,
        history=None,
        max_iterations=DEFAULT_MAX_ITERATIONS,
    ):
        history_text = format_history(history)

        if history_text:
            prompt = f"{history_text}\n\nNew message: {user_input}"
        else:
            prompt = user_input

        current_time = datetime.now().astimezone().isoformat()

        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\nCurrent date and time: {current_time}",
            },
            {"role": "user", "content": prompt},
        ]

        # Tracks the most recent item the conversation actually looked
        # up, so a picture can ride along with the text reply when one
        # exists -- "show the picture when the owner asks about an
        # item" needs *some* signal back to the HTTP layer beyond the
        # plain text, since the image itself has to be fetched
        # separately by the web UI.
        last_item_image = None

        for iteration in range(max_iterations):
            response = chat(
                "responder",
                self.model,
                messages,
                tools=TOOL_SCHEMAS,
                logger=self.logger,
            )

            messages.append(response.message)

            tool_calls = response.message.tool_calls

            if not tool_calls:
                return {
                    "reply": strip_image_paths(
                        strip_markdown(response.message.content or "")
                    ),
                    "image_path": last_item_image,
                }

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments

                result = self.tool_manager.execute(tool_name, arguments)

                if result.get("success"):
                    output = result.get("output")

                    if tool_name == "check_stock" and isinstance(output, dict):
                        last_item_image = output.get("image_path") or last_item_image
                    elif (
                        tool_name == "search_items"
                        and isinstance(output, list)
                        and len(output) == 1
                    ):
                        last_item_image = output[0].get("image_path") or last_item_image

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": str(result),
                    }
                )

        return {
            "reply": (
                "I wasn't able to finish looking into that -- "
                "could you rephrase or try again?"
            ),
            "image_path": last_item_image,
        }
