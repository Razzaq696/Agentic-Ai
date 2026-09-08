"""
Intelligent Task Execution Agent - Environment Health Check
Phase 0: Environment Audit & Project Foundation

Verifies runtime environment, installed packages, project structure, and Ollama service.
Distinguishes between PASS, WARNING, and FAIL.
"""

import sys
import json
import warnings
import urllib.request
import urllib.error
from pathlib import Path

# Suppress known non-critical third-party deprecation notices during health check import
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Required directory structure relative to project root
EXPECTED_DIRS = ["agent", "tools", "ui", "config", "tests"]
EXPECTED_FILES = ["app.py", "requirements.txt", "README.md", ".gitignore", ".env.example"]

# Required packages to test import
REQUIRED_PACKAGES = [
    ("langchain", "LangChain"),
    ("langchain_core", "LangChain Core"),
    ("langchain_community", "LangChain Community"),
    ("langchain_ollama", "LangChain Ollama"),
    ("langgraph", "LangGraph"),
    ("streamlit", "Streamlit"),
    ("dotenv", "Python Dotenv"),
    ("pydantic", "Pydantic"),
]


class HealthCheckRunner:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pass_count = 0
        self.warning_count = 0
        self.fail_count = 0

    def record_pass(self, category: str, message: str) -> None:
        self.pass_count += 1
        print(f"  [PASS]    {category:<18}: {message}")

    def record_warning(self, category: str, message: str) -> None:
        self.warning_count += 1
        print(f"  [WARNING] {category:<18}: {message}")

    def record_fail(self, category: str, message: str) -> None:
        self.fail_count += 1
        print(f"  [FAIL]    {category:<18}: {message}")

    def check_python(self) -> None:
        print("\n1. Python Runtime Environment")
        version_info = sys.version_info
        version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        executable = sys.executable

        if version_info.major == 3 and version_info.minor >= 10:
            self.record_pass("Python Version", f"Python {version_str} ({executable})")
        else:
            self.record_fail("Python Version", f"Python {version_str} (Requires Python 3.10+)")

    def check_package_imports(self) -> None:
        print("\n2. Package Dependencies & Imports")
        for module_name, display_name in REQUIRED_PACKAGES:
            try:
                mod = __import__(module_name)
                version = getattr(mod, "__version__", "installed")
                self.record_pass(display_name, f"v{version}")
            except ImportError as e:
                self.record_fail(display_name, f"Import error: {e}")

    def check_project_structure(self) -> None:
        print("\n3. Project Directory Structure")
        print(f"  Project Root: {self.project_root}")

        # Check required files
        for fname in EXPECTED_FILES:
            file_path = self.project_root / fname
            if file_path.is_file():
                self.record_pass(f"File: {fname}", "Found")
            else:
                self.record_fail(f"File: {fname}", f"Missing file at {file_path}")

        # Check required directories
        for dname in EXPECTED_DIRS:
            dir_path = self.project_root / dname
            init_file = dir_path / "__init__.py"
            if dir_path.is_dir() and init_file.is_file():
                self.record_pass(f"Dir: {dname}/", f"Package directory with __init__.py")
            elif dir_path.is_dir():
                self.record_warning(f"Dir: {dname}/", f"Directory exists but missing __init__.py")
            else:
                self.record_fail(f"Dir: {dname}/", f"Missing directory at {dir_path}")

    def check_ollama_service(self) -> None:
        print("\n4. Ollama Service & Models")
        ollama_url = "http://localhost:11434/api/tags"
        try:
            req = urllib.request.Request(ollama_url, headers={"User-Agent": "HealthCheck/1.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    models = data.get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    self.record_pass("Ollama Service", f"Online at {ollama_url}")

                    # Check for preferred Qwen model
                    qwen_models = [m for m in model_names if "qwen" in m.lower()]
                    if qwen_models:
                        self.record_pass("Qwen Model", f"Available: {', '.join(qwen_models)}")
                    else:
                        self.record_warning("Qwen Model", f"No Qwen model found. Available: {model_names}")
                else:
                    self.record_warning("Ollama Service", f"Returned status HTTP {response.status}")
        except urllib.error.URLError as e:
            self.record_warning(
                "Ollama Service",
                f"Cannot reach Ollama at {ollama_url} ({e.reason}). Local LLM calls will fail if Ollama is stopped.",
            )
        except Exception as e:
            self.record_warning("Ollama Service", f"Error checking Ollama: {e}")

    def run(self) -> int:
        print("=" * 70)
        print("  INTELLIGENT TASK EXECUTION AGENT - HEALTH CHECK (PHASE 0)")
        print("=" * 70)

        self.check_python()
        self.check_package_imports()
        self.check_project_structure()
        self.check_ollama_service()

        print("\n" + "=" * 70)
        print(f"  SUMMARY: {self.pass_count} PASSED | {self.warning_count} WARNINGS | {self.fail_count} FAILED")
        print("=" * 70)

        if self.fail_count == 0:
            print("  OVERALL STATUS: HEALTHY (PASS)\n")
            return 0
        else:
            print("  OVERALL STATUS: UNHEALTHY (FAIL)\n")
            return 1


def main() -> None:
    # Determine project root (same directory as this script or current working directory)
    script_dir = Path(__file__).resolve().parent
    runner = HealthCheckRunner(script_dir)
    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
