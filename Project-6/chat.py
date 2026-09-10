"""Interactive terminal chat interface for Project 6 LangGraph Agent Assistant."""

import sys
from agent.runner import run_agent


def main():
    print("\n" + "=" * 70)
    print("🤖 TELEGRAM AGENTIC AI ASSISTANT — LIVE INTERACTIVE CHAT")
    print("Type your message below and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! 👋\n")
                break

            print("\n[Agent Thinking & Processing...]")
            state = run_agent(user_input)

            route = state.get("selected_action", "UNKNOWN")
            reply = state.get("final_response", "No response generated.")

            print(f"[Route: {route}]")
            print(f"Bot: {reply}\n")
            print("-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n[Error]: {str(e)}\n")


if __name__ == "__main__":
    main()
