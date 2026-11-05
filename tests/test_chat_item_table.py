"""
Which lookups produce the item table under a chat reply.

The table is what the owner taps to change a price or stock, so a question
that matches exactly one product must produce one too (it used to return
only that product's picture). The LLM and the inventory tools are replaced
by fakes: no network, no database.

Run from the project root:
    .venv/bin/python3 -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import responder as responder_module


def tool_call(name, **arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def llm_turn(*calls, content=None):
    message = SimpleNamespace(content=content, tool_calls=list(calls) or None)
    return SimpleNamespace(message=message)


def item(name, image_path=None):
    return {
        "name": name,
        "category": "Generators",
        "current_quantity": 3,
        "cost_price": 100,
        "sale_price": 150,
        "image_path": image_path,
    }


class ChatItemTableTests(unittest.TestCase):
    def run_reply(self, llm_turns, tool_outputs):
        """Runs one respond() with scripted LLM turns and scripted tool results."""
        responder = responder_module.Responder()
        outputs = iter(tool_outputs)

        with mock.patch.object(responder_module, "chat", side_effect=llm_turns), mock.patch.object(
            responder.tool_manager,
            "execute",
            side_effect=lambda name, arguments: {"success": True, "output": next(outputs)},
        ):
            return responder.respond("price of petrol generator")

    def test_single_search_match_gets_a_table_and_keeps_its_picture(self):
        result = self.run_reply(
            [llm_turn(tool_call("search_items", query="petrol generator")), llm_turn(content="done")],
            [[item("Petrol Generator", "product_images/abc.jpg")]],
        )

        self.assertEqual([i["name"] for i in result["items"]], ["Petrol Generator"])
        self.assertEqual(result["image_path"], "product_images/abc.jpg")

    def test_check_stock_of_one_product_gets_a_table(self):
        result = self.run_reply(
            [llm_turn(tool_call("check_stock", item="Petrol Generator")), llm_turn(content="done")],
            [{**item("Petrol Generator"), "recent_transactions": []}],
        )

        self.assertEqual([i["name"] for i in result["items"]], ["Petrol Generator"])

    def test_several_matches_still_get_one_table(self):
        result = self.run_reply(
            [llm_turn(tool_call("search_items", query="generator")), llm_turn(content="done")],
            [[item("Petrol Generator"), item("Diesel Generator")]],
        )

        self.assertEqual(len(result["items"]), 2)

    def test_no_match_gets_no_table(self):
        result = self.run_reply(
            [llm_turn(tool_call("search_items", query="rocket")), llm_turn(content="done")],
            [[]],
        )

        self.assertIsNone(result["items"])

    def test_latest_lookup_wins(self):
        result = self.run_reply(
            [
                llm_turn(tool_call("search_items", query="generator")),
                llm_turn(tool_call("check_stock", item="Diesel Generator")),
                llm_turn(content="done"),
            ],
            [
                [item("Petrol Generator"), item("Diesel Generator")],
                {**item("Diesel Generator"), "recent_transactions": []},
            ],
        )

        self.assertEqual([i["name"] for i in result["items"]], ["Diesel Generator"])

    def test_empty_second_lookup_keeps_the_earlier_table(self):
        result = self.run_reply(
            [
                llm_turn(tool_call("search_items", query="generator")),
                llm_turn(tool_call("search_items", query="rocket")),
                llm_turn(content="done"),
            ],
            [[item("Petrol Generator")], []],
        )

        self.assertEqual([i["name"] for i in result["items"]], ["Petrol Generator"])


if __name__ == "__main__":
    unittest.main()
