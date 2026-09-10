"""Environment and Setup Verification Test for Phase 0."""

import sys
import os
import platform
import importlib.metadata
import unittest
from pathlib import Path


class TestEnvironment(unittest.TestCase):
    """Test suite for validating the environment setup and project structure."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent

    def test_01_python_version(self):
        """Verify Python version is 3.10 or higher."""
        major, minor = sys.version_info[:2]
        self.assertGreaterEqual(
            (major, minor),
            (3, 10),
            f"Python 3.10+ required, found Python {major}.{minor}"
        )
        print(f"\n[PASS] Python version: {sys.version.split()[0]} on {platform.system()} {platform.release()}")

    def test_02_project_structure(self):
        """Verify the required project directories and files exist."""
        required_paths = [
            "app.py",
            "requirements.txt",
            "README.md",
            ".env.example",
            ".gitignore",
            "data",
            "rag/__init__.py",
            "agent/__init__.py",
            "config/__init__.py",
            "tests",
        ]
        for rel_path in required_paths:
            full_path = self.project_root / rel_path
            self.assertTrue(
                full_path.exists(),
                f"Missing required path: {rel_path}"
            )
        print("[PASS] Project structure is intact.")

    def test_03_package_imports(self):
        """Verify all required packages are importable and display versions."""
        packages = {
            "langchain": "langchain",
            "langchain-core": "langchain_core",
            "langchain-community": "langchain_community",
            "langchain-ollama": "langchain_ollama",
            "langgraph": "langgraph",
            "chromadb": "chromadb",
            "streamlit": "streamlit",
            "pydantic": "pydantic",
            "python-dotenv": "dotenv",
        }

        for pkg_name, module_name in packages.items():
            mod = __import__(module_name)
            self.assertIsNotNone(mod, f"Failed to import {module_name}")
            version = importlib.metadata.version(pkg_name)
            self.assertTrue(bool(version), f"Could not determine version for {pkg_name}")
            print(f"[PASS] {pkg_name} ({version}) imported successfully.")

    def test_04_ollama_connectivity_and_model(self):
        """Verify Ollama is reachable and the local Qwen model responds."""
        from langchain_ollama import ChatOllama

        model_name = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        llm = ChatOllama(model=model_name, temperature=0.0)
        response = llm.invoke("Respond with the single word: OK")
        self.assertTrue(
            bool(response.content),
            "Received empty response from Ollama model."
        )
        print(f"[PASS] Ollama model ({model_name}) responded: '{response.content.strip()}'")

    def test_05_chroma_minimal_init(self):
        """Verify Chroma can initialize an ephemeral client."""
        import chromadb

        client = chromadb.EphemeralClient()
        collection_name = "test_phase0_probe"
        collection = client.create_collection(name=collection_name)
        self.assertEqual(collection.name, collection_name)
        self.assertEqual(client.count_collections(), 1)
        client.delete_collection(name=collection_name)
        print("[PASS] Chroma EphemeralClient initialized and verified.")

    def test_06_langgraph_minimal_init(self):
        """Verify LangGraph StateGraph can be initialized and compiled."""
        from typing import TypedDict
        from langgraph.graph import StateGraph, START, END

        class ProbeState(TypedDict):
            stage: str

        builder = StateGraph(ProbeState)
        builder.add_node("step1", lambda s: {"stage": "ready"})
        builder.add_edge(START, "step1")
        builder.add_edge("step1", END)
        graph = builder.compile()

        result = graph.invoke({"stage": "init"})
        self.assertEqual(result, {"stage": "ready"})
        print("[PASS] LangGraph StateGraph compiled and executed test step.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
