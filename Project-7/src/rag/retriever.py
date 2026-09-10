"""Requirements-aware product retriever using Chroma vector search."""

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

from src.config import settings
from src.models.schemas import BudgetInfo, RetrievedProduct
from src.models.state import ShoppingState
from src.rag.vectorstore import ChromaVectorStoreManager
from src.utils.logger import logger


class ProductRetriever:
    """Retrieves relevant products from Chroma based on structured shopping requirements."""

    def __init__(
        self,
        vector_store_manager: Optional[ChromaVectorStoreManager] = None,
        top_k: Optional[int] = None,
    ):
        """Initialize retriever.

        Args:
            vector_store_manager: ChromaVectorStoreManager instance. Defaults to singleton instance.
            top_k: Number of products to retrieve. Defaults to settings.rag_top_k.
        """
        self.vector_store_manager = vector_store_manager or ChromaVectorStoreManager()
        self.top_k = top_k or settings.rag_top_k

    def construct_query(
        self,
        category: Optional[str],
        requirements: List[str],
        preferences: List[str],
        intended_use: Optional[str],
    ) -> str:
        """Construct an expressive semantic query from structured requirement elements."""
        query_parts: List[str] = []
        if category:
            query_parts.append(category)
        if requirements:
            query_parts.extend(requirements)
        if preferences:
            query_parts.extend(preferences)
        if intended_use and intended_use != "General / specified use case":
            query_parts.append(intended_use)

        query = " ".join(query_parts).strip()
        logger.debug(f"Constructed retrieval query: {query!r}")
        return query

    def retrieve(
        self,
        category: Optional[str] = None,
        budget: Optional[BudgetInfo] = None,
        requirements: Optional[List[str]] = None,
        preferences: Optional[List[str]] = None,
        priorities: Optional[List[str]] = None,
        intended_use: Optional[str] = None,
        custom_query: Optional[str] = None,
    ) -> List[RetrievedProduct]:
        """Retrieve relevant products matching the provided requirements.

        Args:
            category: Identified product category.
            budget: Structured BudgetInfo.
            requirements: Mandatory specifications.
            preferences: Nice-to-have preferences.
            priorities: Decision trade-offs.
            intended_use: Intended persona or use case.
            custom_query: Optional override query string (e.g. for refined searches).

        Returns:
            List of structured RetrievedProduct objects.
        """
        reqs = requirements or []
        prefs = preferences or []

        # Step 1: Construct search query
        search_query = custom_query or self.construct_query(
            category=category,
            requirements=reqs,
            preferences=prefs,
            intended_use=intended_use,
        )

        if not search_query:
            logger.warning("Empty search query generated; returning empty retrieval list")
            return []

        # Step 2: Attempt category-filtered search if category is established
        filter_dict = None
        if category:
            filter_dict = {"category": category}

        docs_with_scores = self.vector_store_manager.similarity_search_with_score(
            query=search_query,
            k=self.top_k,
            filter_dict=filter_dict,
        )

        # Fallback to unfiltered search if filtered search returned 0 items
        if not docs_with_scores and filter_dict:
            logger.info(
                f"No results found with category filter '{category}'. Retrying without category filter."
            )
            docs_with_scores = self.vector_store_manager.similarity_search_with_score(
                query=search_query,
                k=self.top_k,
                filter_dict=None,
            )

        if not docs_with_scores:
            logger.info(f"No products retrieved for query: '{search_query}'")
            return []

        # Step 3: Format and annotate results into RetrievedProduct models
        retrieved_products: List[RetrievedProduct] = []
        max_budget = budget.max_amount if (budget and budget.is_specified) else None

        for doc, score in docs_with_scores:
            metadata = doc.metadata or {}
            p_price = float(metadata.get("price", 0.0))
            p_category = metadata.get("category", "")
            p_name = metadata.get("product_name", "Unknown Product")

            # Evaluate budget fit
            budget_notes = []
            if max_budget is not None:
                if p_price <= max_budget:
                    budget_notes.append(f"Within budget (${p_price:,.2f} <= ${max_budget:,.2f})")
                else:
                    budget_notes.append(f"Exceeds target budget (${p_price:,.2f} > ${max_budget:,.2f})")

            match_notes_str = "; ".join(budget_notes) if budget_notes else "Budget flexible"

            # Parse features and specifications from page content
            parsed_specs: dict = {}
            parsed_features: List[str] = []

            for line in doc.page_content.splitlines():
                if line.startswith("Key Specifications:"):
                    specs_part = line[len("Key Specifications:"):].strip()
                    if specs_part and specs_part != "None":
                        for pair in specs_part.split(", "):
                            if ": " in pair:
                                sk, sv = pair.split(": ", 1)
                                parsed_specs[sk.strip()] = sv.strip()
                elif line.startswith("Notable Features:"):
                    feats_part = line[len("Notable Features:"):].strip()
                    if feats_part and feats_part != "None":
                        parsed_features = [f.strip() for f in feats_part.split(", ") if f.strip()]

            # Also match requested features against page content
            for r in reqs:
                if r.lower() in doc.page_content.lower() and r not in parsed_features:
                    parsed_features.append(r)

            # Similarity score normalization (Chroma score is L2 distance where lower is closer)
            normalized_score = max(0.0, min(1.0, 1.0 / (1.0 + float(score))))

            retrieved = RetrievedProduct(
                product_id=str(metadata.get("product_id", "")),
                product_name=p_name,
                category=p_category,
                brand=metadata.get("brand", ""),
                price=p_price,
                currency=metadata.get("currency", "USD"),
                specifications=parsed_specs,
                features=parsed_features,
                source=metadata.get("source", "Local Knowledge Base"),
                similarity_score=round(normalized_score, 4),
                match_notes=match_notes_str,
            )
            retrieved_products.append(retrieved)

        logger.info(f"Retrieved {len(retrieved_products)} products for query: '{search_query}'")
        return retrieved_products

    def retrieve_for_state(
        self,
        state: ShoppingState,
        custom_query: Optional[str] = None,
    ) -> List[RetrievedProduct]:
        """Convenience method to retrieve products directly from LangGraph ShoppingState."""
        budget_dict = state.get("budget") or {}
        budget_info = BudgetInfo(**budget_dict) if budget_dict else None

        return self.retrieve(
            category=state.get("product_category"),
            budget=budget_info,
            requirements=state.get("requirements") or [],
            preferences=state.get("preferences") or [],
            priorities=state.get("priorities") or [],
            intended_use=state.get("intended_use"),
            custom_query=custom_query or state.get("retrieval_query"),
        )
