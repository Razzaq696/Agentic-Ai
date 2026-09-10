import pytest
from streamlit.testing.v1 import AppTest

def test_streamlit_empty_input():
    """Verify that clicking Run Agents with empty input displays warning message."""
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    
    at.text_area[0].input("   ").run()
    at.button[0].click().run()
    
    assert len(at.warning) > 0
    assert "Please enter a problem first." in at.warning[0].value

def test_streamlit_ui_elements():
    """Verify core UI components exist."""
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert len(at.title) > 0
    assert "Multi-Agent Problem Solving System" in at.title[0].value
    assert len(at.text_area) > 0
    assert len(at.button) > 0
    assert at.button[0].label == "Run Agents"

def test_streamlit_execution_workflow():
    """Verify executing problem through Streamlit app."""
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception

    at.text_area[0].input("Calculate the total cost of 15 items at $24 each and explain the result.").run()
    at.button[0].click().run()

    assert not at.exception
    assert len(at.markdown) > 0
    markdown_texts = [m.value for m in at.markdown]
    has_360 = any("360" in t for t in markdown_texts)
    assert has_360
