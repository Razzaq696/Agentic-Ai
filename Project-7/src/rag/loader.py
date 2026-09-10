"""Product dataset loading, validation, and document preparation."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from pydantic import ValidationError

from src.config import settings
from src.models.schemas import ProductRecord
from src.utils.logger import logger


def load_product_dataset(
    file_path: Optional[str] = None,
) -> Tuple[List[ProductRecord], List[str]]:
    """Load and validate product records from a JSON dataset.

    Args:
        file_path: Optional path to JSON file. Defaults to settings.product_dataset_path.

    Returns:
        Tuple of (list_of_valid_product_records, list_of_error_messages).
    """
    target_path = file_path or settings.product_dataset_path
    logger.info(f"Loading product dataset from: {target_path}")

    if not os.path.exists(target_path):
        err = f"Product dataset file not found at: {target_path}"
        logger.error(err)
        return [], [err]

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as jde:
        err = f"Malformed JSON in product dataset: {jde}"
        logger.error(err)
        return [], [err]
    except Exception as e:
        err = f"Failed to read product dataset: {e}"
        logger.error(err)
        return [], [err]

    if not isinstance(raw_data, list):
        err = f"Dataset root must be a JSON array, got {type(raw_data).__name__}"
        logger.error(err)
        return [], [err]

    valid_products: List[ProductRecord] = []
    errors: List[str] = []

    for idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            errors.append(f"Item #{idx} is not a valid dictionary object")
            continue
        try:
            record = ProductRecord.model_validate(item)
            valid_products.append(record)
        except ValidationError as ve:
            p_name = item.get("product_name", f"item #{idx}")
            err_msg = f"Validation failure for '{p_name}': {ve.errors()[0]['msg']}"
            logger.warning(err_msg)
            errors.append(err_msg)

    logger.info(
        f"Successfully loaded {len(valid_products)} product records ({len(errors)} invalid items skipped)"
    )
    return valid_products, errors


def product_to_text(product: ProductRecord) -> str:
    """Format a product record into a rich, searchable textual representation."""
    specs_str = ", ".join(f"{k}: {v}" for k, v in product.specifications.items()) or "None"
    features_str = ", ".join(product.features) or "None"
    use_cases_str = ", ".join(product.use_cases) or "General"
    pros_str = "; ".join(product.pros) or "N/A"
    cons_str = "; ".join(product.cons) or "N/A"

    return (
        f"Product: {product.product_name}\n"
        f"Brand: {product.brand}\n"
        f"Category: {product.category}\n"
        f"Price: ${product.price:,.2f} {product.currency}\n"
        f"Key Specifications: {specs_str}\n"
        f"Notable Features: {features_str}\n"
        f"Best Use Cases: {use_cases_str}\n"
        f"Pros: {pros_str}\n"
        f"Cons: {cons_str}\n"
        f"Rating: {product.rating or 'Unrated'}/5.0\n"
        f"Source Reference: {product.source}"
    )


def prepare_documents(products: List[ProductRecord]) -> List[Document]:
    """Convert valid ProductRecord objects into LangChain Documents with metadata."""
    documents: List[Document] = []
    for p in products:
        text_content = product_to_text(p)
        metadata = {
            "product_id": p.product_id,
            "product_name": p.product_name,
            "brand": p.brand,
            "category": p.category,
            "price": float(p.price),
            "currency": p.currency,
            "rating": float(p.rating) if p.rating is not None else 0.0,
            "source": p.source,
        }
        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)

    logger.debug(f"Prepared {len(documents)} searchable documents with metadata")
    return documents
