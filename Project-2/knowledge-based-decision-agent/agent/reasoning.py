"""Agent Reasoning and Decision/Answer generation node."""

import re
from typing import Any, Dict, List
from agent.state import AgentState
from agent.llm import get_llm


def format_context_for_prompt(context_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved context chunks into structured text for the LLM prompt."""
    if not context_chunks:
        return "No relevant policy documents found."

    formatted_parts = []
    for idx, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "Unknown")
        title = chunk.get("title", "")
        content = chunk.get("content", "").strip()
        formatted_parts.append(
            f"[Document {idx}] Source: {source} | Title: {title}\n{content}"
        )
    return "\n\n".join(formatted_parts)


def is_out_of_domain(query: str) -> bool:
    """Detect if a query is clearly out of domain or unsupported by the knowledge base."""
    q_lower = query.lower()
    
    # Common non-academic out-of-domain terms
    out_of_domain_keywords = [
        "fifa", "world cup", "football", "soccer", "president of", "capital of",
        "weather", "movie", "celebrity", "crypto", "bitcoin", "recipe", "cooking"
    ]
    return any(k in q_lower for k in out_of_domain_keywords)


def generate_reasoning_prompt(query: str, formatted_context: str) -> str:
    """Construct the grounding prompt for the decision agent."""
    return (
        "You are a Knowledge-Based Decision Agent for university student services.\n"
        "Answer the user query, make recommendations, or evaluate eligibility decisions using ONLY the provided Knowledge Base Context below.\n\n"
        "RULES:\n"
        "1. Base your answer STRICTLY on the facts in the Knowledge Base Context.\n"
        "2. If evaluating eligibility (e.g. GPA requirement), explicitly state if the student is eligible or not based on the policy threshold.\n"
        "3. If providing recommendations (e.g. attendance percentage), advise on the required policy steps (e.g. medical excuses or Dean petitions).\n"
        "4. If the context does NOT contain enough information, state: 'The requested information is not available in the university knowledge base.'\n"
        "5. Do NOT invent policies, numbers, or requirements.\n\n"
        f"KNOWLEDGE BASE CONTEXT:\n{formatted_context}\n\n"
        f"USER QUERY:\n{query}\n\n"
        "AGENT DECISION / ANSWER:"
    )


def reason_and_decide(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: Reasons over retrieved context to generate a grounded answer, decision, or recommendation.

    Input: state["query"], state["context"]
    Output: state update with "answer" and "sources"

    Args:
        state: Current AgentState dictionary.

    Returns:
        State update dictionary containing "answer" and "sources".
    """
    query = state.get("query", "").strip()
    context_chunks = state.get("context", [])

    if not query:
        return {
            "answer": "Please provide a valid question or query.",
            "sources": [],
            "error": "Empty query received."
        }

    # Check for unsupported or out-of-domain queries
    if not context_chunks or is_out_of_domain(query):
        return {
            "answer": "The requested information is not available in the university knowledge base.",
            "sources": [],
        }

    # Identify unique sources present in retrieved context
    unique_sources = sorted(list({c.get("source") for c in context_chunks if c.get("source")}))
    formatted_context = format_context_for_prompt(context_chunks)
    prompt = generate_reasoning_prompt(query, formatted_context)

    try:
        llm = get_llm(temperature=0.0, num_predict=200)
        response = llm.invoke(prompt)
        answer_text = response.content.strip()

        # Clean up any residual think tags if model outputs them
        answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()

        if not answer_text:
            answer_text = "The requested information is not available in the university knowledge base."

        return {
            "answer": answer_text,
            "sources": unique_sources,
        }
    except Exception as e:
        print(f"[Warning] LLM reasoning exception ({e}), generating grounded fallback.")
        top_chunk = context_chunks[0]
        fallback_answer = f"Based on {top_chunk.get('source', 'university policy')}:\n{top_chunk.get('content', '')}"
        return {
            "answer": fallback_answer,
            "sources": unique_sources,
            "error": f"LLM warning: {str(e)}"
        }


if __name__ == "__main__":
    test_state: AgentState = {
        "query": "What are the attendance requirements for students?",
        "retrieval_query": "attendance requirements for students",
        "context": [
            {
                "source": "attendance_policy.txt",
                "title": "University Student Attendance Policy",
                "content": "All registered undergraduate and graduate students must maintain a minimum attendance rate of 80% in lectures, discussions, tutorials, and practical laboratory sessions."
            }
        ],
        "answer": "",
        "sources": [],
        "error": None
    }
    res = reason_and_decide(test_state)
    print("Generated Answer:")
    print(res["answer"])
    print("\nSources:", res["sources"])
