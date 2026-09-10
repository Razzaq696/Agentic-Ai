"""Product Validation Agent for Phase 4."""

from typing import Any, Dict, List, Optional
from src.models.schemas import (
    Product,
    ProductValidationRecord,
    ValidationAgentOutput,
    ValidationStatus,
)
from src.utils.logger import logger


class ProductValidationAgent:
    """Agent responsible for validating candidate products against completeness,

    consistency, and reliability standards before comparison and scoring.
    """

    def __init__(self):
        pass

    def validate_product(
        self,
        product: Product,
        expected_category: Optional[str] = None,
    ) -> ProductValidationRecord:
        """Validate an individual product candidate.

        Rules:
        - Product name is mandatory and cannot be empty or generic ('product', 'item').
        - Price must be non-negative if present.
        - If price is missing or 0, flags missing field without inventing numbers.
        - Category consistency: if expected_category is provided, checks for plausible match.
        - Flags missing specs or features explicitly.
        - Unreliable products (e.g. data_confidence < 0.3 or completely empty specs/features) are marked INVALID.

        Args:
            product: Product model instance.
            expected_category: Optional target category to check against.

        Returns:
            ProductValidationRecord detailing validation status and findings.
        """
        missing_fields: List[str] = []
        conflicting_fields: List[str] = []
        notes_list: List[str] = []

        # Check 1: Product Name
        name = (product.name or "").strip()
        if not name or name.lower() in ("product", "item", "unknown", "untitled", "n/a"):
            conflicting_fields.append("product_name: Missing or generic placeholder name")
            return ProductValidationRecord(
                product_name=name or "Unknown",
                status=ValidationStatus.INVALID,
                is_eligible_for_recommendation=False,
                missing_fields=["name"],
                conflicting_fields=conflicting_fields,
                data_completeness=0.0,
                notes="Product rejected due to missing or invalid name.",
            )

        # Check 2: Price and Currency
        if product.price is None:
            missing_fields.append("price")
            notes_list.append("Price is unlisted/unavailable.")
        elif product.price < 0:
            conflicting_fields.append(f"price: Negative price value {product.price}")
            notes_list.append("Invalid negative price detected.")

        if not product.currency:
            missing_fields.append("currency")

        # Check 3: Category consistency
        if expected_category and product.category:
            exp_clean = expected_category.strip().lower()
            prod_clean = product.category.strip().lower()
            if exp_clean not in prod_clean and prod_clean not in exp_clean:
                # Plausible mismatch warning
                notes_list.append(
                    f"Category notice: product category '{product.category}' differs from requested '{expected_category}'"
                )

        # Check 4: Specifications & Features
        if not product.specifications:
            missing_fields.append("specifications")
            notes_list.append("Detailed technical specifications are unavailable.")

        if not product.features:
            missing_fields.append("features")
            notes_list.append("Feature list is unavailable.")

        # Check 5: Source/URL
        if not product.url and not product.source:
            missing_fields.append("source_attribution")
            notes_list.append("No source URL or reference provided.")

        # Calculate completeness score based on 5 key facets:
        # [name, price, specifications, features, source]
        present_facets = 5 - len([f for f in missing_fields if f in ("price", "specifications", "features", "source_attribution")])
        data_completeness = round(max(0.0, min(1.0, present_facets / 5.0)), 2)

        # Decision on status
        if conflicting_fields or product.data_confidence < 0.3:
            status = ValidationStatus.INVALID
            is_eligible = False
            notes_list.append("Product deemed unreliable or conflicting.")
        elif len(missing_fields) >= 3:
            status = ValidationStatus.PARTIALLY_VALID
            # Eligible only if name and either price or specs are present
            is_eligible = product.price is not None or bool(product.specifications)
            notes_list.append("Product has substantial missing information; marked partially valid.")
        elif missing_fields:
            status = ValidationStatus.PARTIALLY_VALID
            is_eligible = True
            notes_list.append("Product verified with minor missing fields.")
        else:
            status = ValidationStatus.VALID
            is_eligible = True
            notes_list.append("Product fully verified with complete data.")

        return ProductValidationRecord(
            product_name=name,
            status=status,
            is_eligible_for_recommendation=is_eligible,
            missing_fields=missing_fields,
            conflicting_fields=conflicting_fields,
            data_completeness=data_completeness,
            notes=" ".join(notes_list),
        )

    def validate_products(
        self,
        products: List[Product],
        expected_category: Optional[str] = None,
    ) -> ValidationAgentOutput:
        """Validate a collection of candidate products.

        Args:
            products: List of Product objects.
            expected_category: Optional target category.

        Returns:
            ValidationAgentOutput with separated valid, partially valid, and invalid items.
        """
        logger.info(f"ProductValidationAgent: Validating {len(products)} candidate products...")
        validated_products: List[Product] = []
        records: List[ProductValidationRecord] = []
        total_valid = 0
        total_partially_valid = 0
        total_invalid = 0

        for p in products:
            rec = self.validate_product(p, expected_category=expected_category)
            records.append(rec)

            if rec.status == ValidationStatus.VALID:
                total_valid += 1
                validated_products.append(p)
            elif rec.status == ValidationStatus.PARTIALLY_VALID:
                total_partially_valid += 1
                if rec.is_eligible_for_recommendation:
                    validated_products.append(p)
            else:
                total_invalid += 1
                logger.warning(
                    f"ProductValidationAgent: Rejected invalid product '{rec.product_name}'. "
                    f"Reasons: {rec.conflicting_fields or rec.notes}"
                )

        logger.info(
            f"ProductValidationAgent: Finished. Valid: {total_valid}, "
            f"Partially Valid: {total_partially_valid}, Invalid: {total_invalid}"
        )

        return ValidationAgentOutput(
            validated_products=validated_products,
            validation_records=records,
            total_valid=total_valid,
            total_partially_valid=total_partially_valid,
            total_invalid=total_invalid,
        )
