"""LangGraph StateGraph workflow construction and compilation."""

from typing import Optional
from langgraph.graph import StateGraph, START, END

from src.agents.comparison_agent import ProductComparisonAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.requirement_agent import RequirementAnalysisAgent
from src.agents.validation_agent import ProductValidationAgent
from src.config import settings
from src.models.schemas import NextAction, RAGDecision
from src.models.state import ShoppingState
from src.integrations.langsmith import configure_langsmith
from src.integrations.n8n_client import N8nClient
from src.integrations.pushover_client import PushoverClient
from src.integrations.sendgrid_client import SendGridClient
from src.rag.evaluator import AgenticRAGEvaluator
from src.rag.loader import load_product_dataset, prepare_documents
from src.rag.retriever import ProductRetriever
from src.rag.vectorstore import ChromaVectorStoreManager
from src.tools.browser_tool import ProductPageExtractorTool
from src.tools.search_tool import WebSearchTool
from src.workflow.nodes import (
    analyze_shopping_request,
    validate_requirements,
    determine_next_action,
    retrieve_product_knowledge,
    evaluate_retrieval,
    decide_research,
    execute_web_search,
    evaluate_search_results,
    extract_product_pages,
    validate_research_results,
    validate_candidate_products_node,
    compare_products_node,
    score_products_node,
    make_decision_node,
    validate_final_decision_node,
    dispatch_integrations_node,
)
from src.utils.logger import logger


def ensure_knowledge_base_indexed(vsm: Optional[ChromaVectorStoreManager] = None) -> int:
    """Ensure the local product knowledge base is indexed in Chroma."""
    manager = vsm or ChromaVectorStoreManager()
    count = manager.get_collection_count()
    if count == 0:
        logger.info("Chroma collection is empty. Auto-indexing local product catalog...")
        products, errors = load_product_dataset()
        if products:
            docs = prepare_documents(products)
            indexed_count = manager.initialize_with_documents(docs)
            logger.info(f"Auto-indexed {indexed_count} products into knowledge base.")
            return indexed_count
    return count


def build_shopping_graph(
    agent: Optional[RequirementAnalysisAgent] = None,
    retriever: Optional[ProductRetriever] = None,
    evaluator: Optional[AgenticRAGEvaluator] = None,
    search_tool: Optional[WebSearchTool] = None,
    browser_tool: Optional[ProductPageExtractorTool] = None,
    validation_agent: Optional[ProductValidationAgent] = None,
    comparison_agent: Optional[ProductComparisonAgent] = None,
    decision_agent: Optional[DecisionAgent] = None,
    n8n_client: Optional[N8nClient] = None,
    sendgrid_client: Optional[SendGridClient] = None,
    pushover_client: Optional[PushoverClient] = None,
):
    """Build and compile the LangGraph workflow for Phases 1–5.

    Workflow topology:
    START
      ↓
    analyze_shopping_request
      ↓
    validate_requirements
      ↓
    determine_next_action
      ├── (if REQUEST_CLARIFICATION or INVALID_INPUT) ──> END
      └── (if PROCEED_TO_PRODUCT_RESEARCH) ──> retrieve_product_knowledge
                                                      ↓
                                              evaluate_retrieval
                                                      ├── (if REFINE_QUERY & retries <= max) ──> retrieve_product_knowledge
                                                      ├── (if retrieval_sufficient == True) ──> validate_candidate_products
                                                      └── (if research_needed == True) ──> decide_research
                                                                                              │
                                                                                              ├── (if skipped) ──> validate_candidate_products
                                                                                              └── (otherwise) ──> execute_web_search
                                                                                                                      ↓
                                                                                                                  evaluate_search_results
                                                                                                                      ↓
                                                                                                                  extract_product_pages
                                                                                                                      ↓
                                                                                                                  validate_research_results
                                                                                                                      ↓
                                                                                                                  validate_candidate_products
                                                                                                                      ↓
                                                                                                                  compare_products
                                                                                                                      ↓
                                                                                                                  score_products
                                                                                                                      ↓
                                                                                                                  make_decision
                                                                                                                      ↓
                                                                                                                  validate_final_decision
                                                                                                                      ↓
                                                                                                                  dispatch_integrations
                                                                                                                      ↓
                                                                                                                     END
    """
    # Initialize LangSmith tracing if enabled in settings
    configure_langsmith()

    workflow = StateGraph(ShoppingState)

    # Custom node wrappers for dependency injection
    def analyze_node(state: ShoppingState):
        return analyze_shopping_request(state, agent=agent)

    def retrieve_node(state: ShoppingState):
        return retrieve_product_knowledge(state, retriever=retriever)

    def evaluate_node(state: ShoppingState):
        return evaluate_retrieval(state, evaluator=evaluator)

    def decide_res_node(state: ShoppingState):
        return decide_research(state, search_tool=search_tool)

    def search_node(state: ShoppingState):
        return execute_web_search(state, search_tool=search_tool)

    def extract_node(state: ShoppingState):
        return extract_product_pages(state, browser_tool=browser_tool)

    def val_prod_node(state: ShoppingState):
        return validate_candidate_products_node(state, agent=validation_agent)

    def comp_prod_node(state: ShoppingState):
        return compare_products_node(state, agent=comparison_agent)

    def dec_node(state: ShoppingState):
        return make_decision_node(state, agent=decision_agent)

    def integrations_node(state: ShoppingState):
        return dispatch_integrations_node(
            state,
            n8n_client=n8n_client,
            sendgrid_client=sendgrid_client,
            pushover_client=pushover_client,
        )

    # Register Nodes
    workflow.add_node("analyze_shopping_request", analyze_node)
    workflow.add_node("validate_requirements", validate_requirements)
    workflow.add_node("determine_next_action", determine_next_action)
    workflow.add_node("retrieve_product_knowledge", retrieve_node)
    workflow.add_node("evaluate_retrieval", evaluate_node)
    workflow.add_node("decide_research", decide_res_node)
    workflow.add_node("execute_web_search", search_node)
    workflow.add_node("evaluate_search_results", evaluate_search_results)
    workflow.add_node("extract_product_pages", extract_node)
    workflow.add_node("validate_research_results", validate_research_results)

    # Phase 4 Multi-Agent Nodes
    workflow.add_node("validate_candidate_products", val_prod_node)
    workflow.add_node("compare_products", comp_prod_node)
    workflow.add_node("score_products", score_products_node)
    workflow.add_node("make_decision", dec_node)
    workflow.add_node("validate_final_decision", validate_final_decision_node)

    # Phase 5 Automation & Notification Integrations Node
    workflow.add_node("dispatch_integrations", integrations_node)

    # Register Sequential Edges
    workflow.add_edge(START, "analyze_shopping_request")
    workflow.add_edge("analyze_shopping_request", "validate_requirements")
    workflow.add_edge("validate_requirements", "determine_next_action")

    # Routing 1: After determine_next_action
    def route_after_action(state: ShoppingState) -> str:
        action = state.get("next_action")
        if action in (NextAction.REQUEST_CLARIFICATION.value, NextAction.INVALID_INPUT.value):
            return END
        return "retrieve_product_knowledge"

    workflow.add_conditional_edges(
        "determine_next_action",
        route_after_action,
        {
            END: END,
            "retrieve_product_knowledge": "retrieve_product_knowledge",
        },
    )

    workflow.add_edge("retrieve_product_knowledge", "evaluate_retrieval")

    # Routing 2: After evaluate_retrieval (Agentic RAG decision loop)
    def route_after_evaluation(state: ShoppingState) -> str:
        decision = state.get("retrieval_decision")
        retry_count = state.get("retrieval_retry_count", 0)

        # Query refinement loop for Chroma RAG
        if decision == RAGDecision.REFINE_QUERY.value and retry_count <= 1:
            return "retrieve_product_knowledge"

        # If Chroma knowledge is sufficient, proceed directly to Product Validation Agent
        if state.get("retrieval_sufficient") is True and not state.get("research_needed"):
            return "validate_candidate_products"

        # If external research is needed, transition to Phase 3 research loop
        return "decide_research"

    workflow.add_conditional_edges(
        "evaluate_retrieval",
        route_after_evaluation,
        {
            "retrieve_product_knowledge": "retrieve_product_knowledge",
            "validate_candidate_products": "validate_candidate_products",
            "decide_research": "decide_research",
        },
    )

    # Routing 3: After decide_research
    def route_after_decide_research(state: ShoppingState) -> str:
        if state.get("research_status") == "skipped":
            return "validate_candidate_products"
        return "execute_web_search"

    workflow.add_conditional_edges(
        "decide_research",
        route_after_decide_research,
        {
            "execute_web_search": "execute_web_search",
            "validate_candidate_products": "validate_candidate_products",
        },
    )

    # Tool execution chain in ReAct loop
    workflow.add_edge("execute_web_search", "evaluate_search_results")
    workflow.add_edge("evaluate_search_results", "extract_product_pages")
    workflow.add_edge("extract_product_pages", "validate_research_results")

    # Routing 4: After validate_research_results -> transitions to Multi-Agent pipeline
    def route_after_research(state: ShoppingState) -> str:
        return "validate_candidate_products"

    workflow.add_conditional_edges(
        "validate_research_results",
        route_after_research,
        {
            "validate_candidate_products": "validate_candidate_products",
        },
    )

    # Phase 4 Multi-Agent Sequential Flow
    workflow.add_edge("validate_candidate_products", "compare_products")
    workflow.add_edge("compare_products", "score_products")
    workflow.add_edge("score_products", "make_decision")
    workflow.add_edge("make_decision", "validate_final_decision")

    # Phase 5 Integrations Flow
    workflow.add_edge("validate_final_decision", "dispatch_integrations")
    workflow.add_edge("dispatch_integrations", END)

    return workflow.compile()


# Default compiled graph ready for execution
create_shopping_workflow = build_shopping_graph
app = build_shopping_graph()


if __name__ == "__main__":
    ensure_knowledge_base_indexed()
    sample_request = "I need an espresso machine with dual boiler and PID control under $2000"
    print(f"\n--- Running Phase 3 Workflow with request: '{sample_request}' ---")
    result = app.invoke({"user_request": sample_request})
    print(f"\nNext Action: {result.get('next_action')}")
    print(f"Product Category: {result.get('product_category')}")
    print(f"Retrieval Sufficient: {result.get('retrieval_sufficient')}")
    print(f"Research Needed: {result.get('research_needed')}")
    print(f"Research Status: {result.get('research_status')}")
    print(f"\nFinal Response:\n{result.get('final_response')}")
