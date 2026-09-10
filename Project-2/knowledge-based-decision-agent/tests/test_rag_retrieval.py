import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest

from rag.loader import load_documents, get_data_directory
from rag.splitter import split_documents
from rag.embeddings import get_embedding_model
from rag.vector_store import build_knowledge_base, get_vector_store, get_default_persist_directory
from rag.retriever import get_retriever, retrieve_context


class TestRagRetrieval(unittest.TestCase):
    """Test suite covering the complete Phase 1 RAG pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = get_data_directory()
        cls.persist_dir = get_default_persist_directory()
        # Build vector store once for the test run
        cls.vector_store = build_knowledge_base(
            data_directory=cls.data_dir,
            persist_directory=cls.persist_dir,
            force_rebuild=True
        )
        cls.retriever = get_retriever(k=3)

    def test_01_documents_load(self):
        """Test 1: Confirm local knowledge-base documents load with expected count and metadata."""
        docs = load_documents(self.data_dir)
        self.assertGreaterEqual(len(docs), 5, "Should load at least 5 policy documents.")
        sources = {d.metadata.get("source") for d in docs}
        self.assertIn("attendance_policy.txt", sources)
        self.assertIn("scholarship_policy.txt", sources)
        self.assertIn("examination_policy.txt", sources)
        print(f"\n[PASS] Loaded {len(docs)} documents successfully.")

    def test_02_chunking(self):
        """Test 2: Confirm documents are split into meaningful chunks with metadata."""
        docs = load_documents(self.data_dir)
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        self.assertGreater(len(chunks), len(docs), "Chunks count should exceed raw documents count.")
        for chunk in chunks:
            self.assertTrue(bool(chunk.page_content), "Chunk content must not be empty.")
            self.assertIn("source", chunk.metadata, "Chunk must retain source metadata.")
            self.assertIn("chunk_index", chunk.metadata, "Chunk must have chunk_index.")
        print(f"[PASS] Successfully generated {len(chunks)} chunks with preserved metadata.")

    def test_03_chroma_indexing(self):
        """Test 3: Confirm chunks are stored in Chroma and collection is populated."""
        store = get_vector_store(self.persist_dir)
        collection = store._collection
        count = collection.count()
        self.assertGreater(count, 0, "Chroma collection count should be greater than 0.")
        print(f"[PASS] Chroma collection is indexed with {count} chunks.")

    def test_04_attendance_retrieval(self):
        """Test 4: Verify attendance query retrieves attendance policy context."""
        query = "What are the attendance requirements and minimum percentage?"
        results = self.retriever.retrieve(query, k=3)
        self.assertGreater(len(results), 0, "Retrieval returned empty results.")
        top_sources = [r["source"] for r in results]
        self.assertIn("attendance_policy.txt", top_sources, "Attendance policy must be retrieved.")
        top_chunk = results[0]
        self.assertEqual(top_chunk["source"], "attendance_policy.txt")
        self.assertTrue("80%" in top_chunk["content"] or "attendance" in top_chunk["content"].lower())
        print(f"[PASS] Attendance query accurately retrieved: {top_sources}")

    def test_05_scholarship_retrieval(self):
        """Test 5: Verify scholarship query retrieves scholarship policy context."""
        query = "Who can apply for the merit scholarship and what GPA is required?"
        results = self.retriever.retrieve(query, k=3)
        self.assertGreater(len(results), 0, "Retrieval returned empty results.")
        top_sources = [r["source"] for r in results]
        self.assertIn("scholarship_policy.txt", top_sources, "Scholarship policy must be retrieved.")
        top_chunk = results[0]
        self.assertEqual(top_chunk["source"], "scholarship_policy.txt")
        self.assertTrue("scholarship" in top_chunk["content"].lower() or "gpa" in top_chunk["content"].lower())
        print(f"[PASS] Scholarship query accurately retrieved: {top_sources}")

    def test_06_examination_retrieval(self):
        """Test 6: Verify examination query retrieves examination policy context."""
        query = "What are the rules for final examinations and make-up exams?"
        results = self.retriever.retrieve(query, k=3)
        self.assertGreater(len(results), 0, "Retrieval returned empty results.")
        top_sources = [r["source"] for r in results]
        self.assertIn("examination_policy.txt", top_sources, "Examination policy must be retrieved.")
        top_chunk = results[0]
        self.assertEqual(top_chunk["source"], "examination_policy.txt")
        self.assertTrue("exam" in top_chunk["content"].lower() or "examination" in top_chunk["content"].lower())
        print(f"[PASS] Examination query accurately retrieved: {top_sources}")

    def test_07_source_metadata(self):
        """Test 7: Verify retrieved chunks retain filename, title, and chunk identifiers."""
        query = "When are semester fees due?"
        results = self.retriever.retrieve(query, k=2)
        self.assertGreater(len(results), 0)
        for res in results:
            self.assertTrue(res["source"].endswith(".txt"), "Source must be a .txt filename.")
            self.assertTrue(bool(res["title"]), "Title metadata must not be empty.")
            self.assertTrue(bool(res["content"]), "Content must not be empty.")
            self.assertIsInstance(res["score"], float, "Score must be a float.")
        print(f"[PASS] Retrieved results contain all required metadata fields.")

    def test_08_reindexing_safety_no_duplicates(self):
        """Test 8: Verify re-indexing is idempotent and does not create duplicate chunks."""
        count_before = self.vector_store._collection.count()
        # Re-run build_knowledge_base
        new_store = build_knowledge_base(
            data_directory=self.data_dir,
            persist_directory=self.persist_dir,
            force_rebuild=True
        )
        count_after = new_store._collection.count()
        self.assertEqual(
            count_before,
            count_after,
            f"Chunk count changed after re-indexing: before={count_before}, after={count_after}"
        )
        print(f"[PASS] Re-indexing safety verified: chunk count remained constant at {count_after}.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
