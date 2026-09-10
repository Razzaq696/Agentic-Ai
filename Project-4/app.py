"""Automated Document Processing System - Premium Streamlit Web Application."""
import os
import time
import json
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd

from src.pipeline import process_document_pipeline
from src.schema import PipelineResult
from src.config import SAMPLE_DOCS_DIR

# -----------------------------------------------------------------------------
# Streamlit Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocFlow AI — Automated Document Processing",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Modern Aesthetics
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(16px);
}

.hero-title {
    font-size: 2.25rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.5;
}

/* Feature Badge */
.feature-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    margin-right: 8px;
    margin-top: 8px;
}

/* Glass Card */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.3);
    margin-bottom: 1.25rem;
}

/* Metric Highlight Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.7) 100%);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(129, 140, 248, 0.4);
}

.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f8fafc;
}

.metric-value.highlight {
    color: #38bdf8;
}

.metric-value.total {
    color: #34d399;
    font-size: 1.45rem;
}

/* Status Badges */
.status-pill-valid {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 8px 18px;
    border-radius: 30px;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    letter-spacing: 0.02em;
}

.status-pill-invalid {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 8px 18px;
    border-radius: 30px;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
    letter-spacing: 0.02em;
}

/* Pipeline Stepper */
.stepper-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 0.85rem 1.25rem;
    margin-bottom: 1.5rem;
}

.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 4px;
}

.step-number {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid #818cf8;
    color: #c7d2fe;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}

.step-label {
    font-size: 0.75rem;
    color: #94a3b8;
    font-weight: 500;
}

.step-arrow {
    color: #475569;
    font-weight: bold;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📄 Automated Document Processing System")

# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-subtitle">
            Enterprise document intelligence powered by <b>LangChain RAG</b>, 
            <b>Chroma Vector Database</b>, <b>Ollama LLM Agent</b>, and <b>Pydantic Type Validation</b> with self-correcting feedback loops.
        </div>
        <div>
            <span class="feature-badge">⚡ LangChain 0.3</span>
            <span class="feature-badge">📦 ChromaDB Local Embeddings</span>
            <span class="feature-badge">🤖 Llama 3.2 Ollama</span>
            <span class="feature-badge">🛡️ Pydantic v2 Schema</span>
            <span class="feature-badge">🔄 Self-Healing Retries</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Pipeline Stepper Visualizer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="stepper-container">
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-label">Document Loader</div>
        </div>
        <div class="step-arrow">➔</div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-label">Chunk Splitter</div>
        </div>
        <div class="step-arrow">➔</div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-label">Chroma DB RAG</div>
        </div>
        <div class="step-arrow">➔</div>
        <div class="step-item">
            <div class="step-number">4</div>
            <div class="step-label">LLM Extractor</div>
        </div>
        <div class="step-arrow">➔</div>
        <div class="step-item">
            <div class="step-number">5</div>
            <div class="step-label">Pydantic Validator</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Main Application Layout: 2 Columns
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

# -----------------------------------------------------------------------------
# LEFT COLUMN: Document Input & Options
# -----------------------------------------------------------------------------
with left_col:
    st.subheader("📥 Document Ingestion")

    # Sample Document Quick Loader
    with st.expander("📂 Load Built-in Sample Invoices (1-Click)", expanded=False):
        c_txt, c_pdf = st.columns(2)
        txt_sample_path = SAMPLE_DOCS_DIR / "sample_invoice.txt"
        pdf_sample_path = SAMPLE_DOCS_DIR / "sample_invoice.pdf"

        if c_txt.button("📄 Load Sample TXT", use_container_width=True):
            if txt_sample_path.exists():
                st.session_state["preset_file"] = str(txt_sample_path)
                st.session_state["preset_name"] = txt_sample_path.name
                st.session_state["preset_bytes"] = txt_sample_path.read_bytes()
                st.rerun()

        if c_pdf.button("📑 Load Sample PDF", use_container_width=True):
            if pdf_sample_path.exists():
                st.session_state["preset_file"] = str(pdf_sample_path)
                st.session_state["preset_name"] = pdf_sample_path.name
                st.session_state["preset_bytes"] = pdf_sample_path.read_bytes()
                st.rerun()

    # File Uploader
    uploaded_file = st.file_uploader(
        "Choose a document to process (TXT or PDF)",
        type=["txt", "pdf"],
        help="Upload a local TXT or PDF document containing invoice or structured text."
    )

    # Resolve active file: either uploaded or preset
    active_bytes = None
    active_name = None

    if uploaded_file is not None:
        active_bytes = uploaded_file.getvalue()
        active_name = uploaded_file.name
        # Clear preset if user uploads fresh file
        if "preset_file" in st.session_state:
            del st.session_state["preset_file"]
    elif "preset_file" in st.session_state:
        active_bytes = st.session_state.get("preset_bytes")
        active_name = st.session_state.get("preset_name")
        st.info(f"Loaded Preset: `{active_name}`")

    # Document details card
    if active_bytes and active_name:
        size_kb = len(active_bytes) / 1024
        file_ext = Path(active_name).suffix.upper().replace(".", "")
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-weight: 700; color: #cbd5e1; margin-bottom: 8px;">📄 Document Selected</div>
                <div style="display: flex; gap: 16px; color: #94a3b8; font-size: 0.9rem;">
                    <span><b>Name:</b> {active_name}</span>
                    <span><b>Format:</b> {file_ext}</span>
                    <span><b>Size:</b> {size_kb:.2f} KB</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Snippet preview if text file
        if active_name.endswith(".txt"):
            with st.expander("👁️ Document Content Preview", expanded=False):
                try:
                    preview_str = active_bytes.decode("utf-8")[:600]
                    st.text(preview_str + ("..." if len(active_bytes) > 600 else ""))
                except Exception:
                    st.write("Preview unavailable.")

    # Execution Button
    process_btn = st.button("🚀 Process Document", key="process_btn", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# RIGHT COLUMN: Extraction & Validation Results
# -----------------------------------------------------------------------------
with right_col:
    st.subheader("📊 Execution & Validation Status")

    if process_btn:
        if not active_bytes or not active_name:
            st.warning("⚠️ Please upload a document before clicking 'Process Document'.")
        else:
            with st.spinner("🤖 Running RAG Retrieval, LLM Extraction & Pydantic Validation..."):
                start_time = time.time()

                # Save file to temporary storage for processing
                suffix = Path(active_name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(active_bytes)
                    tmp_path = tmp_file.name

                try:
                    # Execute Existing Phase 1 Backend Pipeline
                    result: PipelineResult = process_document_pipeline(tmp_path)
                    elapsed_time = time.time() - start_time

                    # Re-processing feedback notification
                    if result.reprocessed:
                        st.info("ℹ️ Initial extraction encountered validation errors. Auto-reprocessing with LLM feedback was executed.")

                    # SUCCESS BRANCH
                    if result.status == "SUCCESS" and result.extracted_data:
                        data = result.extracted_data
                        st.success("✅ Validation Status: Valid")

                        # Header Status Badge & Metrics
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem;">
                                <div class="status-pill-valid">
                                    <span>✔</span> Validation Status: Valid
                                </div>
                                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500;">
                                    ⏱️ Execution Time: <b>{elapsed_time:.2f}s</b> | Attempts: <b>{result.attempts}</b>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Structured Result Tabs
                        tab_summary, tab_json, tab_rag = st.tabs([
                            "📋 Structured Invoice",
                            "🔍 Pydantic JSON",
                            "📑 Retrieved RAG Chunks"
                        ])

                        with tab_summary:
                            # Metric highlight grid
                            m1, m2 = st.columns(2)
                            with m1:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Document Type</div>
                                        <div class="metric-value">{data.document_type}</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">Invoice Number</div>
                                        <div class="metric-value highlight">{data.invoice_number}</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">Invoice Date</div>
                                        <div class="metric-value">{data.date}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            with m2:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Customer Name</div>
                                        <div class="metric-value">{data.customer_name}</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">Grand Total Due</div>
                                        <div class="metric-value total">${data.total_amount:,.2f}</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-label">Line Items Count</div>
                                        <div class="metric-value">{len(data.items)} Items</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            # Line items table
                            st.markdown("#### 🛒 Line Items Breakdown")
                            items_data = [
                                {
                                    "Item #": idx + 1,
                                    "Description": item.description,
                                    "Quantity": item.quantity,
                                    "Unit Price": f"${item.unit_price:,.2f}",
                                    "Total Price": f"${item.total:,.2f}"
                                }
                                for idx, item in enumerate(data.items)
                            ]
                            df_items = pd.DataFrame(items_data)
                            st.dataframe(df_items, hide_index=True, use_container_width=True)

                        with tab_json:
                            st.markdown("#### 🛡️ Validated Pydantic Schema Model")
                            st.json(data.model_dump())
                            json_str = json.dumps(data.model_dump(), indent=2)
                            st.download_button(
                                label="💾 Download Structured JSON",
                                data=json_str,
                                file_name=f"{data.invoice_number}_extracted.json",
                                mime="application/json",
                                use_container_width=True
                            )

                        with tab_rag:
                            st.markdown("#### 📑 Chroma DB Retrieved Chunks")
                            for idx, chunk in enumerate(result.retrieved_chunks, 1):
                                st.markdown(f"**Chunk #{idx}**")
                                st.code(chunk.strip(), language="text")

                    # FAILED BRANCH
                    else:
                        st.markdown(
                            """
                            <div class="status-pill-invalid" style="margin-bottom: 1.25rem;">
                                <span>✖</span> Validation Status: Invalid
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.error("❌ Document extraction failed Pydantic validation after all allowed retries.")
                        st.markdown("#### ⚠️ Validation Diagnostics")
                        for err in result.errors:
                            st.error(f"• {err}")

                        if result.retrieved_chunks:
                            with st.expander("📑 View Retrieved Chunks"):
                                for i, chunk in enumerate(result.retrieved_chunks, 1):
                                    st.code(chunk, language="text")

                except Exception as ex:
                    st.error(f"Pipeline Execution Error: {str(ex)}")
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
    else:
        st.info("Upload a document on the left and click **🚀 Process Document** to start.")
