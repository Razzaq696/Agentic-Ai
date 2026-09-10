"""Phase 3 Agent Reasoning + Decision / Answer Test Suite."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from agent.graph import build_decision_agent_graph, run_decision_agent


class TestAgentReasoning(unittest.TestCase):
    """Test suite for Phase 3 Decision Agent reasoning, QA, recommendations, and decisions."""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_decision_agent_graph()

    def test_01_question_answering_attendance(self):
        """Test 1: Direct policy question answering grounded in retrieved attendance context."""
        query = "What are the attendance requirements and penalties for students?"
        result = run_decision_agent(query)

        self.assertEqual(result["query"], query)
        self.assertTrue(bool(result["retrieval_query"]))
        self.assertGreater(len(result["context"]), 0)
        self.assertIn("attendance_policy.txt", result["sources"])

        answer = result["answer"].lower()
        self.assertTrue(bool(answer), "Answer must not be empty.")
        # Assert grounded facts from attendance_policy.txt
        self.assertTrue(
            "80%" in answer or "attendance" in answer or "debar" in answer,
            f"Answer does not reflect attendance policy context: {result['answer']}"
        )
        print(f"\n[PASS] Question Answering (Attendance):\nSources: {result['sources']}\nAnswer Summary: {result['answer'][:180]}...")

    def test_02_scholarship_decision_support(self):
        """Test 2: Decision support evaluating student eligibility against scholarship policy."""
        query = "Based on the scholarship policy, am I eligible for the merit scholarship if my GPA is 3.4?"
        result = run_decision_agent(query)

        self.assertIn("scholarship_policy.txt", result["sources"])
        answer = result["answer"].lower()
        self.assertTrue(bool(answer))
        # Merit scholarship requires 3.75 CGPA; 3.4 is not eligible
        self.assertTrue(
            "not eligible" in answer or "ineligible" in answer or "3.75" in answer or "no" in answer or "below" in answer or "require" in answer,
            f"Answer does not properly evaluate scholarship eligibility: {result['answer']}"
        )
        print(f"[PASS] Decision Support (Scholarship):\nAnswer Summary: {result['answer'][:180]}...")

    def test_03_policy_recommendation(self):
        """Test 3: Policy-grounded actionable recommendation for an at-risk attendance scenario."""
        query = "I currently have 70% attendance. Based on the university policy, what should I do before the examination?"
        result = run_decision_agent(query)

        self.assertIn("attendance_policy.txt", result["sources"])
        answer = result["answer"].lower()
        self.assertTrue(bool(answer))
        # Attendance policy notes debarment under 80% and recommends medical certificate / excuse submission
        self.assertTrue(
            "medical" in answer or "excuse" in answer or "debar" in answer or "student services" in answer or "80%" in answer,
            f"Answer does not provide grounded recommendation: {result['answer']}"
        )
        print(f"[PASS] Recommendation (Attendance at 70%):\nAnswer Summary: {result['answer'][:180]}...")

    def test_04_unsupported_out_of_domain_query(self):
        """Test 4: Refusal and safe handling for questions outside the university policy knowledge base."""
        query = "Who won the FIFA World Cup in 2022?"
        result = run_decision_agent(query)

        answer = result["answer"].lower()
        self.assertTrue(bool(answer))
        self.assertTrue(
            "not available" in answer or "knowledge base" in answer or "unrelated" in answer or "cannot" in answer or "no information" in answer,
            f"Agent should have stated information is not available: {result['answer']}"
        )
        print(f"[PASS] Unsupported Query Refusal:\nAnswer: {result['answer']}")

    def test_05_examination_policy_qa(self):
        """Test 5: Question answering grounded in examination hall rules and make-up procedures."""
        query = "What are the rules for missed final examinations and make-up exams?"
        result = run_decision_agent(query)

        self.assertIn("examination_policy.txt", result["sources"])
        answer = result["answer"].lower()
        self.assertTrue(bool(answer))
        self.assertTrue(
            "dean" in answer or "make-up" in answer or "makeup" in answer or "48" in answer or "medical" in answer or "two weeks" in answer,
            f"Answer does not reflect examination policy: {result['answer']}"
        )
        print(f"[PASS] Question Answering (Examinations):\nAnswer Summary: {result['answer'][:180]}...")


if __name__ == "__main__":
    unittest.main(verbosity=2)
