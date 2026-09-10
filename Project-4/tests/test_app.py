"""Streamlit App integration tests using Streamlit AppTest framework."""
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from src.config import SAMPLE_DOCS_DIR


@pytest.fixture
def sample_txt_path():
    return SAMPLE_DOCS_DIR / "sample_invoice.txt"


def test_app_initial_render():
    """Verify that the Streamlit app renders initial state cleanly."""
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert "Automated Document Processing System" in at.title[0].value
    process_btn = at.button(key="process_btn")
    assert process_btn is not None
    assert "Process Document" in process_btn.label


def test_app_empty_input_warning():
    """Verify that clicking Process Document without a file triggers a warning."""
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception

    # Click Process Document without uploading
    at.button(key="process_btn").click().run()
    assert not at.exception
    assert len(at.warning) >= 1
    assert "Please upload a document" in at.warning[0].value


def test_app_processing_pipeline(sample_txt_path):
    """Verify that uploading a document and clicking process executes pipeline successfully."""
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception

    # Upload file
    with open(sample_txt_path, "rb") as f:
        file_bytes = f.read()

    at.file_uploader[0].upload("sample_invoice.txt", file_bytes).run()
    assert not at.exception

    # Click Process Document
    at.button(key="process_btn").click().run()
    assert not at.exception

    # Verify success output
    assert len(at.success) >= 1
    assert "Valid" in at.success[0].value
