# pricing.py
#
# Static USD-per-million-token pricing, used to estimate cost from the
# token counts the API returns with every response. This is NOT pulled
# live from OpenRouter -- it's a local table that can drift from their
# actual list price over time, so treat it as a running estimate good
# enough to catch "did something unexpectedly expensive happen", not
# as a substitute for checking the real bill at openrouter.ai/activity.
#
# Add a model's pricing here when you switch to it. An unlisted model
# still gets its token counts logged -- cost just comes back as None
# rather than a silently wrong guess.

MODEL_PRICING_USD_PER_MILLION = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "qwen2.5:1.5b": {"input": 0.0, "output": 0.0},  # local via Ollama -- free
}


def estimate_cost(model, prompt_tokens, completion_tokens):
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model)

    if not pricing or prompt_tokens is None or completion_tokens is None:
        return None

    return (
        prompt_tokens * pricing["input"] / 1_000_000
        + completion_tokens * pricing["output"] / 1_000_000
    )
