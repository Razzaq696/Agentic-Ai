"""Project 6 Phase 2: LangGraph Agent Assistant Demo."""

from agent.runner import run_agent


def print_step(title: str, content: str, color: str = "\033[94m"):
    reset = "\033[0m"
    print(f"\n{color}==> {title}{reset}")
    print(content)


def demonstrate_query(label: str, query: str):
    print("\n" + "=" * 75)
    print(f"[{label.upper()}] INPUT QUERY: \"{query}\"")
    print("=" * 75)

    state = run_agent(query)

    print_step("1. Query Analysis", f"Detected Intent: {state.get('query_analysis')}")
    print_step("2. Selected Action", f"Routing Decision: {state.get('selected_action')}")

    if state.get("retrieved_context"):
        print_step("3. RAG Retrieved Context", "\n".join(f"• {c}" for c in state.get("retrieved_context", [])))

    if state.get("tool_result"):
        print_step("3. Tool Execution Payload", str(state.get("tool_result")))

    print_step("4. Processed Result (Normalized)", str(state.get("processed_result")))
    print_step("5. Validation Status", f"Status: {state.get('validation_status')}")
    print_step("6. Final Reply Generated", f"{state.get('final_response')}", color="\033[92m")

    if state.get("error"):
        print_step("Error / Notice Encountered", f"{state.get('error')}", color="\033[91m")


def main():
    print("\n" + "#" * 75)
    print("# PROJECT 6: TELEGRAM AGENTIC AI ASSISTANT (COMPLETE WORKFLOW)")
    print("# LangGraph Agent + LLM Reasoning + RAG Knowledge + Tools/APIs + Validation")
    print("#" * 75)


    test_cases = [
        ("Test 1: LLM Reasoning Route", "Explain why Python is commonly used in AI development."),
        ("Test 2: RAG Knowledge Route", "What are the core features and knowledge base of the Telegram Agentic AI Assistant?"),
        ("Test 3: Tool Calculator Route", "Calculate 125 × 8."),
        ("Test 4: Tool Division by Zero Failure", "Calculate 10 / 0"),
        ("Test 5: Empty Input Validation Failure", ""),
    ]

    for label, q in test_cases:
        demonstrate_query(label, q)

    print("\n" + "#" * 75)
    print("# ALL COMPLETE WORKFLOW DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("#" * 75)



if __name__ == "__main__":
    main()
