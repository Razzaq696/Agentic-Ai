"""
Unit and Integration Test for Phase 1 LLM Connection.
Verifies ChatOllama initialization, Ollama reachability, and Qwen3:4B invocation.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm import get_llm
from langchain_core.messages import AIMessage


class TestLLMConnection(unittest.TestCase):
    def setUp(self):
        self.llm = get_llm()

    def test_llm_initialization(self):
        """Verify ChatOllama initializes with configured parameters."""
        self.assertIsNotNone(self.llm)
        self.assertEqual(self.llm.model, "qwen3:4b")

    def test_llm_response(self):
        """Verify that Ollama responds to a prompt via LangChain."""
        test_prompt = "Explain what an AI agent is in one sentence."
        response = self.llm.invoke(test_prompt)

        # Check response structure
        self.assertIsInstance(response, AIMessage)
        self.assertIsNotNone(response.content)
        self.assertTrue(len(str(response.content).strip()) > 0)
        print(f"\n[Test Prompt]: {test_prompt}")
        print(f"[LLM Response]: {response.content.strip()}\n")


if __name__ == "__main__":
    unittest.main()
