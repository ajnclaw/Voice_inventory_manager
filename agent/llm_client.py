"""
llm_client.py

Single choke point for every LLM call in this codebase (Responder).
Talks to whichever provider is active in config.LLM_PROVIDERS (Ollama /
DeepSeek / OpenRouter, or another added there later) over its
OpenAI-compatible /chat/completions endpoint, using plain HTTP (stdlib
only, no new dependency). Ported unchanged from ~/ai-agent's
llm_client.py -- this part of that codebase is fully domain-agnostic.

    response.message.content
    response.message.tool_calls[i].function.name
    response.message.tool_calls[i].function.arguments   (a dict, not a
                                                           JSON string)

That shape is also dict-subscriptable (response["message"]["content"]).
"""

import json
import urllib.error
import urllib.request

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_PROVIDER


class AttrDict(dict):
    """A dict that's also readable via attribute access (both
    response.message and response["message"] work)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def _serialize_messages(messages):
    """
    Convert this codebase's message list into what an OpenAI-compatible
    API strictly requires:

    1. An assistant tool_calls entry needs its arguments JSON-encoded as
       a *string*, not passed as a live dict.
    2. A "tool" role message needs a "tool_call_id" tying it back to the
       specific call it answers (and "name", not this codebase's
       "tool_name"). Reconstructed positionally: responder.py always
       appends exactly one tool-role message per tool_call, in the same
       order the tool_calls appeared in the preceding assistant turn.
    """
    serialized = []
    pending_call_ids = []

    for message in messages:
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            tool_calls_out = []
            pending_call_ids = []

            for index, call in enumerate(message["tool_calls"]):
                call_id = call.get("id") or f"call_{index}"
                pending_call_ids.append(call_id)

                function = call["function"]
                arguments = function.get("arguments", {})
                arguments_str = (
                    arguments if isinstance(arguments, str) else json.dumps(arguments)
                )

                tool_call_out = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": arguments_str,
                    },
                }

                # Gemini-specific passthrough: its OpenAI-compat endpoint
                # attaches an extra_content.google.thought_signature to
                # each function call and then *requires* that exact value
                # echoed back on the same call in the next request, or it
                # rejects the turn with a 400 (see
                # https://ai.google.dev/gemini-api/docs/thought-signatures).
                # Other providers never set this field, so this is a no-op
                # for them.
                if call.get("extra_content"):
                    tool_call_out["extra_content"] = call["extra_content"]

                tool_calls_out.append(tool_call_out)

            serialized.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls_out,
                }
            )

        elif role == "tool":
            call_id = pending_call_ids.pop(0) if pending_call_ids else None

            serialized.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": message.get("tool_name") or message.get("name"),
                    "content": message.get("content", ""),
                }
            )

        else:
            serialized.append(
                {
                    "role": role,
                    "content": message.get("content", ""),
                }
            )

    return serialized


def _parse_tool_calls(raw_tool_calls):
    tool_calls = []

    for call in raw_tool_calls or []:
        function = call.get("function", {})
        raw_arguments = function.get("arguments", "{}")

        try:
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else (raw_arguments or {})
            )
        except json.JSONDecodeError:
            arguments = {}

        tool_calls.append(
            AttrDict(
                {
                    "id": call.get("id"),
                    "type": call.get("type", "function"),
                    "function": AttrDict(
                        {
                            "name": function.get("name"),
                            "arguments": arguments,
                        }
                    ),
                    # See the matching comment in _serialize_messages --
                    # carried through unchanged so it can be echoed back
                    # on Gemini's next request. Absent/None for every
                    # other provider.
                    "extra_content": call.get("extra_content"),
                }
            )
        )

    return tool_calls


def _parse_response(raw):
    choice = (raw.get("choices") or [{}])[0]
    raw_message = choice.get("message", {})

    message = AttrDict(
        {
            "role": raw_message.get("role", "assistant"),
            "content": raw_message.get("content"),
            "tool_calls": _parse_tool_calls(raw_message.get("tool_calls")),
        }
    )

    # Every OpenAI-compatible response includes token counts -- this
    # was being silently discarded before, which meant no cost/usage
    # tracking was possible anywhere in the codebase no matter what
    # logger.py did with it.
    usage = AttrDict(raw.get("usage") or {})

    return AttrDict({"message": message, "usage": usage})


def chat(component, model, messages, tools=None, logger=None):
    if not LLM_API_KEY:
        raise RuntimeError(
            f"No API key configured for LLM_PROVIDER={LLM_PROVIDER!r} "
            f"(checked the real environment and the project's .env file)."
        )

    payload = {
        "model": model,
        "messages": _serialize_messages(messages),
        "stream": False,
    }

    if tools is not None:
        payload["tools"] = tools

    request = urllib.request.Request(
        f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as http_response:
            raw = json.loads(http_response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{LLM_PROVIDER} API error {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{LLM_PROVIDER} API unreachable: {exc.reason}") from exc

    response = _parse_response(raw)

    if logger:
        logger.log_llm_call(component, model, response, request_messages=messages)

    return response
