# AI Smart Shopping Decision Agent (Project 7)

An enterprise-grade, autonomous **Multi-Agent Decision System** built with **LangGraph**, **Chroma Vector Database (Agentic RAG)**, **Playwright headless research**, **ReAct tool calling**, **deterministic mathematical scoring**, **fault-isolated external integrations (n8n, SendGrid, Pushover, LangSmith)**, and a modern **Streamlit** user interface.

---

## 1. Problem Statement
Online shopping is burdened by fragmented product catalogs, conflicting consumer reviews, deceptive affiliate promotions, and cognitive overload. Consumers frequently struggle to:
- Translate informal requirements into technical specifications.
- Evaluate trade-offs between competing products transparently.
- Cross-reference local catalog data with real-time web pricing and specs.
- Obtain objective, hallucination-free purchase recommendations without forced winners.

---

## 2. Target Users
- **Everyday Consumers**: Seeking unbiased, objective product recommendations tailored to specific functional needs and budget caps.
- **Power Users & Professionals**: Researching technical purchases (e.g. espresso gear, coding keyboards, developer laptops) requiring strict specification compliance.
- **E-Commerce & Procurement Teams**: Needing automated research, competitive benchmarking, and structured JSON decision dispatches to downstream systems.

---

## 3. Solution Overview
The **AI Smart Shopping Decision Agent** provides an end-to-end autonomous decision pipeline. Rather than generating simple conversational text, the system:
1. Analyzes and structures raw consumer inputs using typed Pydantic models.
2. Checks a local Chroma vector database using semantic similarity and metadata filtering.
3. Automatically triggers external web search and Playwright browser extraction when local knowledge is insufficient.
4. Executes a Multi-Agent evaluation pipeline (Product Validation, Comparative Matrix Analysis, Deterministic Scoring, and Decision Synthesis).
5. Dispatches notifications and webhook events (n8n, SendGrid, Pushover) with full secret isolation.
6. Delivers an interactive, transparent Streamlit dashboard for real-time progress, comparison tables, and source citations.

---

## 4. Key Features
- **Deterministic Requirement Parsing**: Identifies category, budget (fixed or flexible), required features, soft preferences, and priorities.
- **Agentic RAG & Chroma Vector Store**: Embeds and indexes local product documents; dynamically decides whether retrieval is sufficient or external research is needed.
- **Live Web Research & Playwright Extraction**: Autonomous Google/DuckDuckGo/Tavily search combined with headless Playwright extraction of real product pages.
- **Bounded ReAct Tool Calling Loop**: Guardrailed against infinite search loops and invalid URLs.
- **Specialized Multi-Agent Pipeline**:
  - *Validation Agent*: Detects corrupt, incomplete, or placeholder product data.
  - *Comparison Agent*: Builds structured requirement match matrices and articulates concrete trade-offs.
  - *Scoring Engine*: Transparent 100-point mathematical scoring algorithm with hard constraint protection.
  - *Decision Agent*: Selects the top recommended product and closest runner-up alternative.
- **No-Forcing Guardrail**: Refuses to recommend an invalid product if none meet the non-negotiable criteria (`decision_status = "no_satisfying_product"`).
- **Production Integrations**: n8n workflow automation, SendGrid email reports, Pushover push notifications, and LangSmith observability.
- **Interactive Streamlit UI**: User-friendly search interface, structured constraint overrides, real-time progress indicators, and verified clickable product links.

---

## 5. Architecture & Workflow Topology

```
START
  ↓
analyze_shopping_request (Requirement Agent)
  ↓
validate_requirements (Guardrails)
  ↓
determine_next_action
  ├── (if REQUEST_CLARIFICATION or INVALID_INPUT) ──> END
  └── (if PROCEED_TO_PRODUCT_RESEARCH) ──> retrieve_product_knowledge (Chroma RAG)
                                                  ↓
                                          evaluate_retrieval (Agentic Evaluator)
                                                  ├── (if REFINE_QUERY & retries <= max) ──> retrieve_product_knowledge
                                                  ├── (if retrieval_sufficient == True) ──> validate_candidate_products
                                                  └── (if research_needed == True) ──> decide_research
                                                                                          │
                                                                                          ├── (if skipped) ──> validate_candidate_products
                                                                                          └── (otherwise) ──> execute_web_search (Tool)
                                                                                                                  ↓
                                                                                                              evaluate_search_results (Guardrails)
                                                                                                                  ↓
                                                                                                              extract_product_pages (Playwright)
                                                                                                                  ↓
                                                                                                              validate_research_results
                                                                                                                  ↓
                                                                                                              validate_candidate_products (Validation Agent)
                                                                                                                  ↓
                                                                                                              compare_products (Comparison Agent)
                                                                                                                  ↓
                                                                                                              score_products (Scoring Engine)
                                                                                                                  ↓
                                                                                                              make_decision (Decision Agent)
                                                                                                                  ↓
                                                                                                              validate_final_decision (Guardrails)
                                                                                                                  ↓
                                                                                                              dispatch_integrations (n8n, SendGrid, Pushover)
                                                                                                                  ↓
                                                                                                                 END
```

---

## 6. Multi-Agent Roles & Responsibilities

| Agent / Component | File | Core Responsibility |
|---|---|---|
| **Requirement Analysis Agent** | `src/agents/requirement_agent.py` | Extracts typed budget, features, and priorities from raw query; handles clarification. |
| **Agentic RAG Evaluator** | `src/rag/evaluator.py` | Evaluates local vector search relevance; routes to query refinement or web research. |
| **Product Validation Agent** | `src/agents/validation_agent.py` | Inspects candidate products; filters corrupted listings, missing prices, or empty names. |
| **Product Comparison Agent** | `src/agents/comparison_agent.py` | Compares candidates against user criteria; identifies evidence-backed matches and weaknesses. |
| **Deterministic Scoring Engine** | `src/workflow/scoring.py` | Calculates transparent scores out of 100 based on hard requirements, budget, and specs. |
| **Decision Agent** | `src/agents/decision_agent.py` | Selects the optimal choice and closest alternative; enforces no-forcing guardrails. |
| **Integration Dispatcher** | `src/workflow/nodes.py` | Delivers structured payloads to n8n, SendGrid, and Pushover in isolated try/except blocks. |

---

## 7. Technologies Used
- **Orchestration**: LangGraph, LangChain Core
- **LLMs Supported**: OpenAI (GPT-4o, GPT-4o-mini), Google Gemini, Groq, Ollama (Local)
- **Vector Database**: ChromaDB (`chromadb`, `langchain-chroma`)
- **Web Automation**: Playwright (Headless Chromium)
- **Data Validation**: Pydantic v2, Pydantic Settings
- **Frontend UI**: Streamlit
- **Automation & Integrations**: n8n webhooks, SendGrid v3 REST API, Pushover mobile push API
- **Observability**: LangSmith
- **Testing**: Pytest (132 unit and integration tests)

---

## 8. Deep Dive into Key Subsystems

### RAG + Chroma
- Local catalog indexed in Chroma (`data/products.json`).
- Semantic vector similarity combined with metadata filtering on product category and brand.
- Supports both persistent disk storage and in-memory ephemeral modes for testing.

### Web Search & Playwright Extraction
- Reusable `WebSearchTool` supporting Mock, Tavily, Serper, and DuckDuckGo providers.
- `ProductPageExtractorTool` leverages Playwright headless browser to load real product URLs, parse CSS selectors, and extract structured technical specifications and pricing.
- Robust timeout protection and user-agent rotation.

### ReAct / Tool Calling Loop
- Dynamic tool orchestration deciding when to perform search vs page extraction.
- Strict bounded iteration limits preventing runaway execution or cyclic calls.
- Deduplication of candidate products by normalized URL and product name.

### Multi-Agent Comparison & Scoring Engine
- Transparent, weighted mathematical breakdown:
  - **Hard Requirements Match**: 40.0 points (steep penalty if non-negotiable criteria missing)
  - **Budget Fit & Compliance**: 25.0 points (products >10% over budget receive 0 budget points and are marked as violators)
  - **Technical Specification Match**: 15.0 points
  - **Feature Match**: 10.0 points
  - **Preferences & Priorities**: 5.0 points
  - **Data Confidence & Source Completeness**: 5.0 points
- Guaranteed that hard constraint violators cannot become false winners.

### Production Integrations (n8n, SendGrid, Pushover, LangSmith)
- **n8n Client**: Builds structured event JSON (`shopping_recommendation_completed`) containing all scores, citations, and product details, sending via standard HTTP POST.
- **SendGrid Client**: Automatically generates dual-format email reports (clean ASCII text and styled HTML cards with confidence badges and citations) delivered via SendGrid REST API.
- **Pushover Client**: Dispatches priority-aware mobile alerts (Priority 0 for top picks, Priority 1 for no-product-found alerts).
- **LangSmith**: Configured via environment variables to record runs, traces, and latency without intrusive code changes.
- **Fault-Isolated**: All integrations run in independent `try/except` blocks; failure or absence of an integration never halts the core shopping recommendation.

---

## 9. Streamlit Application Overview

The Streamlit app (`app.py`) provides an intuitive dashboard for interactive shopping:
- **Search Controls Sidebar**:
  - Displays Chroma local knowledge base collection size.
  - Optional constraints: Budget cap ($ USD), category hint, preferences, and priorities.
  - Optional integration overrides: toggle n8n webhook, SendGrid email recipient, Pushover user key.
- **Preset Quick-Fills**: One-click example queries (Keyboard, Espresso Machine, Headphones).
- **Dynamic Execution Status**: Step-by-step progress tracking with `st.status`.
- **Structured Results (Sections A through G)**:
  - **Section A**: Interpreted Requirements (Detected Category, Budget limit, Mandatory specs).
  - **Section B**: Product Research Results (Verified cards with specifications and verified URLs).
  - **Section C**: Multi-Agent Comparison (Strengths, weaknesses, and trade-off summary).
  - **Section D**: Transparent Scoring (Interactive DataFrame with sub-score breakdowns).
  - **Section E**: Top Recommendation (Why chosen, key advantages, considerations, confidence badge).
  - **Section F**: Closest Alternative (Runner-up option with comparative pricing).
  - **Section G**: External Automations & Notifications (Delivery status badges).

---

## 10. Project Structure

```
PROJECT 7/
├── app.py                       # Streamlit web application entry point
├── requirements.txt             # Project dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Production Git ignore rules
├── README.md                    # Comprehensive documentation
├── data/
│   ├── products.json            # Local product catalog dataset (14 items)
│   └── chroma_db/               # Persistent Chroma vector store
├── src/
│   ├── config.py                # Pydantic Settings & environment loader
│   ├── models/
│   │   ├── schemas.py           # Pydantic schemas (Validation, Comparison, Scoring, Decision, Integrations)
│   │   └── state.py             # LangGraph ShoppingState TypedDict
│   ├── llm/
│   │   ├── factory.py           # Multi-provider LLM & Embeddings factories (with DeterministicMockEmbeddings)
│   │   └── prompts.py           # Guardrailed system prompts and templates
│   ├── agents/
│   │   ├── requirement_agent.py # Requirement Analysis Agent (Phase 1)
│   │   ├── validation_agent.py  # Product Validation Agent (Phase 4)
│   │   ├── comparison_agent.py  # Product Comparison Agent (Phase 4)
│   │   └── decision_agent.py    # Decision Agent (Phase 4)
│   ├── rag/
│   │   ├── loader.py            # Dataset loading and document preparation
│   │   ├── vectorstore.py       # Chroma vector store manager
│   │   ├── retriever.py         # Requirements-aware product retriever
│   │   └── evaluator.py         # Agentic RAG sufficiency evaluator
│   ├── tools/
│   │   ├── search_tool.py       # Reusable WebSearchTool (Mock, Tavily, Serper, DuckDuckGo)
│   │   ├── browser_tool.py      # Playwright ProductPageExtractorTool
│   │   └── registry.py          # ToolRegistry and function calling schemas
│   ├── integrations/
│   │   ├── n8n_client.py        # n8n Webhook client & payload builder
│   │   ├── sendgrid_client.py   # SendGrid REST client & HTML report formatter
│   │   ├── pushover_client.py   # Pushover mobile push alert client
│   │   └── langsmith.py         # LangSmith tracing synchronization
│   ├── guardrails/
│   │   └── validators.py        # Input, output, RAG, research, and decision guardrails
│   ├── workflow/
│   │   ├── nodes.py             # LangGraph workflow nodes (Phases 1–5)
│   │   ├── scoring.py           # Centralized deterministic scoring engine
│   │   └── graph.py             # LangGraph StateGraph builder & CLI runner
│   ├── ui/
│   │   ├── __init__.py          # UI module export
│   │   └── helpers.py           # Streamlit data extractors, formatters & sanitizers
│   └── utils/
│       └── logger.py            # Structured logging
└── tests/
    ├── conftest.py              # Pytest fixtures and mock factories
    ├── test_guardrails.py       # Guardrail unit tests
    ├── test_models.py           # Pydantic schema validation tests
    ├── test_rag_data.py         # Catalog loading tests
    ├── test_vectorstore.py      # Chroma lifecycle and search tests
    ├── test_retriever.py        # Requirements-based retrieval tests
    ├── test_rag_evaluator.py    # Agentic RAG routing tests
    ├── test_rag_workflow.py     # End-to-end RAG workflow tests
    ├── test_requirement_agent.py# Requirement agent tests
    ├── test_search_tool.py      # Search tool tests
    ├── test_browser_tool.py     # Playwright extraction tests
    ├── test_react_loop.py       # ReAct tool-calling loop tests
    ├── test_phase3_workflow.py  # Phase 3 workflow tests
    ├── test_validation_agent.py # Validation agent tests
    ├── test_comparison_scoring.py# Comparison & scoring tests
    ├── test_decision_agent.py   # Decision agent tests
    ├── test_phase4_workflow.py  # Phase 4 workflow tests
    ├── test_phase5_integrations.py# n8n, SendGrid, Pushover, LangSmith tests
    ├── test_phase6_streamlit.py # Streamlit UI helpers & end-to-end UI tests
    └── test_workflow.py         # Phase 1 regression tests
```

---

## 11. Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Playwright Chromium browser binaries installed

### 2. Clone & Install Dependencies
```bash
# Navigate to project directory
cd "PROJECT 7"

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser engines
playwright install chromium
```

### 3. Configure Environment Variables
Copy the `.env.example` template:
```bash
cp .env.example .env
```

Edit `.env` to configure your API keys:
```ini
# LLM Providers (default is mock mode if empty)
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Web Search
SEARCH_PROVIDER=mock  # Options: mock, tavily, serper, duckduckgo
TAVILY_API_KEY=

# Optional LangSmith Tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=shopping-decision-agent

# Optional n8n Webhook
N8N_ENABLED=false
N8N_WEBHOOK_URL=

# Optional SendGrid Email
SENDGRID_ENABLED=false
SENDGRID_API_KEY=
SENDGRID_FROM_EMAIL=noreply@shopping-agent.ai
SENDGRID_TO_EMAIL=buyer@example.com

# Optional Pushover Push Notifications
PUSHOVER_ENABLED=false
PUSHOVER_USER_KEY=
PUSHOVER_API_TOKEN=
```

---

## 12. How to Run

### Run Streamlit Web Application
```bash
streamlit run app.py
```
The browser will open automatically at `http://localhost:8501`.

### Run CLI Workflow Directly
```bash
python -m src.workflow.graph
```

---

## 13. How to Use the Application

1. **Enter Requirements**: Type your natural-language request into the text area (e.g. *"I need a wireless mechanical keyboard with tactile switches for coding under $250"*).
2. **Set Optional Constraints**: Expand the sidebar to enforce an explicit budget cap, target category, or personal preferences.
3. **Configure Notifications (Optional)**: In the sidebar, check SendGrid or Pushover to receive real-time copies of your recommendations.
4. **Click 'Find Best Product'**: Watch the agent execute the 8-step pipeline in real time.
5. **Review Results**:
   - Inspect the **Interpreted Requirements** to verify that budget and specs were correctly parsed.
   - Expand candidate products to review **technical specifications** and **verified URLs**.
   - Check the **Comparison Matrix** and **Transparent Scoring Table** to understand exact point distributions.
   - Read the **Top Recommendation** card for the grounded selection justification and trade-offs.
   - Review the **Runner-up Alternative** if you wish to consider a secondary option.

---

## 14. Error Handling & Guardrails

| Risk / Failure Mode | Defense / Guardrail |
|---|---|
| **Empty or Nonsensical Input** | Input length and alphabetic checks reject invalid queries with user guidance. |
| **Ambiguous Shopping Query** | Automatically triggers `REQUEST_CLARIFICATION` state with targeted clarifying questions. |
| **Missing Budget** | Defaulted to flexible market pricing without blocking the search workflow. |
| **Local Catalog Mismatch** | Agentic RAG Evaluator detects low similarity and initiates targeted web research. |
| **Malicious or Dead URLs** | URL validator blocks localhost/private IPs and ensures valid web protocols. |
| **Scraper Failure / Timeout** | Playwright runs under explicit timeouts, safely returning partial data with error notes. |
| **Corrupted Product Data** | Product Validation Agent screens out items with negative prices, empty titles, or null specs. |
| **Forced False Winners** | No-Forcing Guardrail marks status as `no_satisfying_product` if hard criteria or budget limits are violated. |
| **External Integration Failure** | Isolated `try/except` wrappers ensure n8n/SendGrid/Pushover errors never break recommendation delivery. |
| **Credential Exposure** | `sanitize_error_message()` actively strips API keys (`sk-...`, `SG...`, `Bearer...`) and local system file paths from UI and error banners. |

---

## 15. Testing & Verification

The project includes **132 automated tests** across 20 test suites:

```bash
# Run the complete test suite
pytest -v
```

### Test Suite Breakdown
- `tests/test_guardrails.py` (14 tests): Input, output, budget, and decision guardrails.
- `tests/test_models.py` (6 tests): Pydantic models, JSON serialization, and field validation.
- `tests/test_rag_data.py` (7 tests): Data loading, text normalization, and malformed catalog handling.
- `tests/test_vectorstore.py` (7 tests): Chroma vector database indexing and similarity retrieval.
- `tests/test_retriever.py` (6 tests): Requirements-aware retrieval and budget filtering.
- `tests/test_rag_evaluator.py` (5 tests): RAG decision sufficiency and query refinement routing.
- `tests/test_rag_workflow.py` (5 tests): LangGraph RAG integration flows.
- `tests/test_requirement_agent.py` (3 tests): Requirement parsing resilience and LLM fallback.
- `tests/test_search_tool.py` (5 tests): Web search query synthesis and provider fallbacks.
- `tests/test_browser_tool.py` (5 tests): Playwright page extraction and timeout protection.
- `tests/test_react_loop.py` (6 tests): ReAct tool calling, iteration capping, and deduplication.
- `tests/test_phase3_workflow.py` (6 tests): Phase 3 web research end-to-end integration.
- `tests/test_validation_agent.py` (5 tests): Candidate product validation and classification.
- `tests/test_comparison_scoring.py` (4 tests): Multi-product comparison matrices and scoring math.
- `tests/test_decision_agent.py` (4 tests): Decision synthesis and no-forcing guardrails.
- `tests/test_phase4_workflow.py` (4 tests): Multi-agent decision workflow execution.
- `tests/test_phase5_integrations.py` (19 tests): n8n, SendGrid, Pushover, LangSmith, and resilience.
- `tests/test_phase6_streamlit.py` (15 tests): UI data extractors, formatters, sanitizers, and UI flows.
- `tests/test_workflow.py` (6 tests): Foundational Phase 1 regression test suite.

**Result**: `132 passed in ~4.8s (100% pass rate)`.

---

## 16. Limitations
- **Dynamic Pricing Fluctuation**: Real-time prices can fluctuate rapidly; prices reflect the latest Playwright extraction or catalog snapshot.
- **Paywalled & Bot-Protected Domains**: Certain retail websites (e.g. Amazon, Best Buy) employ aggressive anti-bot captchas; the agent falls back to open product review and specification hubs.
- **Single-Turn Session Flow**: The current workflow operates on a single-request basis with clarification prompts rather than an unbounded multi-turn conversation memory.

---

## 17. Future Improvements
- **Automated Price History Tracking**: Periodic price monitoring with Pushover alerts upon price drops.
- **Multi-Store Affiliate & Coupon Aggregator**: Automatic discovery of active promo codes during Playwright extraction.
- **User Preference Profiles**: Long-term memory storing preferred brands, sizing, and hardware ecosystems across sessions.
- **Visual Product Comparison**: Screenshot capture and image embedding in the Streamlit comparison tab.

---

## 18. License
MIT License. Built for advanced educational and agentic engineering demonstrations.
