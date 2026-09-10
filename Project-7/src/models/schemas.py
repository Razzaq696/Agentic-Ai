"""Pydantic models for structured requirements, budget extraction, and actions."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class NextAction(str, Enum):
    """Possible next actions determined by the agent workflow."""

    PROCEED_TO_PRODUCT_RESEARCH = "proceed_to_product_research"
    REQUEST_CLARIFICATION = "request_clarification"
    INVALID_INPUT = "invalid_input"


class BudgetInfo(BaseModel):
    """Structured representation of extracted budget constraints."""

    min_amount: Optional[float] = Field(
        default=None,
        description="Minimum budget limit if specified (e.g., 500 for '$500-$1000')",
    )
    max_amount: Optional[float] = Field(
        default=None,
        description="Maximum budget ceiling (e.g., 1200 for 'under $1200')",
    )
    currency: str = Field(
        default="USD",
        description="Currency code or symbol (e.g., USD, EUR, GBP, $)",
    )
    is_flexible: bool = Field(
        default=True,
        description="True if the user indicates willingness to stretch or no hard limit was stated",
    )
    is_specified: bool = Field(
        default=False,
        description="True if an explicit numeric or clear monetary budget was provided",
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Original text describing the budget, if any",
    )

    @field_validator("min_amount", "max_amount", mode="before")
    @classmethod
    def parse_amount(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v) if v >= 0 else None
        if isinstance(v, str):
            clean_v = v.replace("$", "").replace(",", "").strip()
            try:
                val = float(clean_v)
                return val if val >= 0 else None
            except ValueError:
                return None
        return None


class RequirementAnalysisOutput(BaseModel):
    """Structured representation of interpreted user shopping requirements."""

    product_category: Optional[str] = Field(
        default=None,
        description="Identified category or product type (e.g., 'Laptop', 'Mechanical Keyboard', 'Noise-Cancelling Headphones'). None if completely indeterminable.",
    )
    budget: BudgetInfo = Field(
        default_factory=BudgetInfo,
        description="Parsed budget constraints. If none specified, is_specified should be False.",
    )
    required_features: List[str] = Field(
        default_factory=list,
        description="Non-negotiable mandatory specifications (e.g., '16GB RAM', 'ANC', 'USB-C charging').",
    )
    preferences: List[str] = Field(
        default_factory=list,
        description="Soft preferences or nice-to-haves (e.g., 'prefer black color', 'lightweight', 'clicky keys').",
    )
    priorities: List[str] = Field(
        default_factory=list,
        description="Ranked or highlighted priorities (e.g., 'battery life over raw power', 'durability', 'portability').",
    )
    intended_use: Optional[str] = Field(
        default=None,
        description="Primary use case or persona (e.g., 'college coding and lightweight gaming', 'office video conferencing').",
    )
    is_clear: bool = Field(
        default=True,
        description="Whether the request provides enough clear signal about what product is being requested.",
    )
    ambiguities_or_missing_info: List[str] = Field(
        default_factory=list,
        description="List of detected ambiguities, conflicting requirements, or critical missing specs needing clarification.",
    )
    summary: str = Field(
        default="",
        description="A concise 1-2 sentence synthesis of the user's shopping requirements.",
    )


# =====================================================================
# PHASE 2: PRODUCT KNOWLEDGE BASE & AGENTIC RAG SCHEMAS
# =====================================================================

class ProductRecord(BaseModel):
    """Pydantic model representing a product in the local knowledge base."""

    product_id: str = Field(..., description="Unique product identifier (e.g. 'laptop-001')")
    product_name: str = Field(..., description="Full descriptive name of the product")
    category: str = Field(..., description="Product category (e.g. 'Laptop', 'Mechanical Keyboard')")
    brand: str = Field(..., description="Brand or manufacturer name")
    price: float = Field(..., ge=0, description="Product price in currency units")
    currency: str = Field(default="USD", description="Currency code (e.g. USD, EUR)")
    specifications: dict = Field(default_factory=dict, description="Detailed hardware/specifications dictionary")
    features: List[str] = Field(default_factory=list, description="List of notable features and specs")
    use_cases: List[str] = Field(default_factory=list, description="Primary intended use cases")
    pros: List[str] = Field(default_factory=list, description="Key advantages and user pros")
    cons: List[str] = Field(default_factory=list, description="Known drawbacks or trade-offs")
    rating: Optional[float] = Field(default=None, ge=0.0, le=5.0, description="Customer or reviewer rating (0-5)")
    source: str = Field(default="Local Knowledge Base", description="Source reference for data attribution")


class RetrievedProduct(BaseModel):
    """Structured representation of a product retrieved via similarity search."""

    product_id: str = Field(..., description="Unique product ID")
    product_name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    brand: str = Field(..., description="Brand name")
    price: float = Field(..., description="Price")
    currency: str = Field(default="USD", description="Currency code")
    specifications: dict = Field(default_factory=dict, description="Hardware specs")
    features: List[str] = Field(default_factory=list, description="Matched features")
    source: str = Field(default="Local Knowledge Base", description="Document source citation")
    similarity_score: Optional[float] = Field(default=None, description="Similarity or relevance score")
    match_notes: Optional[str] = Field(default=None, description="Summary notes on how this product matches requirements")


class RAGDecision(str, Enum):
    """Possible outcomes from Agentic RAG evaluation."""

    SUFFICIENT = "sufficient"
    REFINE_QUERY = "refine_query"
    EXTERNAL_RESEARCH_NEEDED = "external_research_needed"


class StructuredRAGOutput(BaseModel):
    """Structured RAG output capturing retrieved products, assessment, and next research decision."""

    retrieved_products: List[RetrievedProduct] = Field(
        default_factory=list,
        description="List of retrieved product records with provenance and scores",
    )
    relevant_information: List[str] = Field(
        default_factory=list,
        description="Synthesized facts, specs, and details extracted from retrieved products",
    )
    retrieval_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in the relevance and completeness of retrieved items (0.0 - 1.0)",
    )
    retrieval_sufficient: bool = Field(
        default=False,
        description="True if retrieved knowledge sufficiently answers the user requirements without needing external web research",
    )
    research_needed: bool = Field(
        default=True,
        description="True if external web research is required in Phase 3 to fill information gaps or find alternatives",
    )
    decision: RAGDecision = Field(
        default=RAGDecision.EXTERNAL_RESEARCH_NEEDED,
        description="Agentic RAG decision: 'sufficient', 'refine_query', or 'external_research_needed'",
    )
    reasoning: str = Field(
        default="",
        description="Explanation detailing why retrieval is sufficient or why external research/refinement is necessary",
    )


# =====================================================================
# PHASE 3: WEB SEARCH, PLAYWRIGHT & PRODUCT RESEARCH SCHEMAS
# =====================================================================

class SearchResultItem(BaseModel):
    """Structured representation of a single web search result snippet."""

    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="Destination URL")
    snippet: str = Field(default="", description="Text preview or snippet")
    source: str = Field(default="web_search", description="Source provider")


class WebSearchOutput(BaseModel):
    """Output structure returned by the web search tool."""

    query: str = Field(..., description="Search query executed")
    results: List[SearchResultItem] = Field(default_factory=list, description="Search hits")
    total_results: int = Field(default=0, description="Count of results returned")
    error: Optional[str] = Field(default=None, description="Error message if search failed")


class ExtractedProductPageData(BaseModel):
    """Product information extracted from an individual product page via Playwright."""

    name: str = Field(..., description="Product name extracted from page")
    brand: Optional[str] = Field(default=None, description="Brand name if identified")
    category: Optional[str] = Field(default=None, description="Product category")
    price: Optional[float] = Field(default=None, ge=0.0, description="Price")
    currency: str = Field(default="USD", description="Currency code")
    specifications: dict = Field(default_factory=dict, description="Extracted hardware specifications")
    features: List[str] = Field(default_factory=list, description="Extracted key features")
    availability: Optional[str] = Field(default=None, description="In stock, Out of stock, Pre-order")
    url: str = Field(..., description="Product page URL")
    source: str = Field(default="product_page", description="Source site or domain")
    data_confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Extraction confidence")
    extraction_error: Optional[str] = Field(default=None, description="Error note if partial failure occurred")


class Product(BaseModel):
    """Structured product model representing verified product data from external research."""

    name: str = Field(..., description="Verified product name")
    brand: Optional[str] = Field(default=None, description="Brand name")
    category: Optional[str] = Field(default=None, description="Product category")
    price: Optional[float] = Field(default=None, description="Product price")
    currency: str = Field(default="USD", description="Currency symbol or code")
    specifications: dict = Field(default_factory=dict, description="Specifications dictionary")
    features: List[str] = Field(default_factory=list, description="List of notable features")
    availability: Optional[str] = Field(default=None, description="Availability status")
    url: Optional[str] = Field(default=None, description="Product URL reference")
    source: str = Field(default="web_research", description="Source origin citation")
    data_confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Data reliability confidence")


class ProductResearchResult(BaseModel):
    """Structured output representing complete external web research and Playwright extraction."""

    products: List[Product] = Field(
        default_factory=list,
        description="List of verified products extracted through web research and Playwright",
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Web search queries executed during the research process",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source URLs and domains consulted during research",
    )
    research_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the gathered external research (0.0 - 1.0)",
    )
    research_complete: bool = Field(
        default=False,
        description="True if external research was executed and concluded",
    )
    research_needed: bool = Field(
        default=False,
        description="True if further research remains necessary",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Log of non-fatal tool or extraction errors encountered during research",
    )


# =====================================================================
# PHASE 4: VALIDATION, COMPARISON, SCORING & DECISION SCHEMAS
# =====================================================================

class ValidationStatus(str, Enum):
    """Validation classification for candidate products."""

    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    INVALID = "invalid"


class ProductValidationRecord(BaseModel):
    """Validation report for an individual product candidate."""

    product_name: str = Field(..., description="Name of the product evaluated")
    status: ValidationStatus = Field(..., description="Classification: valid, partially_valid, or invalid")
    is_eligible_for_recommendation: bool = Field(
        default=True,
        description="Whether this product can be considered for the final recommendation",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Explicitly identified missing fields (never hallucinated)",
    )
    conflicting_fields: List[str] = Field(
        default_factory=list,
        description="Inconsistencies or conflicting values detected",
    )
    data_completeness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Proportion of expected fields present (0.0 - 1.0)",
    )
    notes: str = Field(default="", description="Detailed validation notes")


class ValidationAgentOutput(BaseModel):
    """Complete output produced by the Product Validation Agent."""

    validated_products: List[Product] = Field(
        default_factory=list,
        description="Products that are valid or partially valid for evaluation",
    )
    validation_records: List[ProductValidationRecord] = Field(
        default_factory=list,
        description="Validation report for each evaluated product candidate",
    )
    total_valid: int = Field(default=0, description="Count of valid products")
    total_partially_valid: int = Field(default=0, description="Count of partially valid products")
    total_invalid: int = Field(default=0, description="Count of rejected invalid products")


class RequirementMatchDetail(BaseModel):
    """Traceable evaluation of a single requirement against product data."""

    requirement: str = Field(..., description="User requirement or specification being evaluated")
    is_hard_requirement: bool = Field(default=True, description="True if non-negotiable hard requirement")
    is_matched: bool = Field(..., description="Whether the product satisfies this requirement")
    supporting_evidence: Optional[str] = Field(
        default=None,
        description="Specific verified product spec or feature providing evidence",
    )


class ProductComparisonItem(BaseModel):
    """Structured comparison data for a single product against user needs."""

    product_name: str = Field(..., description="Product name")
    brand: Optional[str] = Field(default=None, description="Product brand")
    price: Optional[float] = Field(default=None, description="Verified price")
    currency: str = Field(default="USD", description="Currency")
    requirement_matches: List[RequirementMatchDetail] = Field(
        default_factory=list,
        description="Traceable requirement match evaluations",
    )
    budget_fit: str = Field(
        default="within_budget",
        description="'within_budget', 'exceeds_budget', or 'flexible'",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Verified advantages directly aligned with user needs",
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Verified trade-offs, missing features, or budget gaps",
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Pertinent product info that was unavailable in the verified data",
    )
    comparison_notes: str = Field(default="", description="Summary comparison commentary")


class ComparisonResult(BaseModel):
    """Structured comparison result across all valid candidate products."""

    comparisons: List[ProductComparisonItem] = Field(
        default_factory=list,
        description="Per-product structured comparisons",
    )
    hard_requirements_summary: str = Field(
        default="",
        description="Overall summary of how candidate products satisfied hard criteria",
    )
    key_tradeoffs: List[str] = Field(
        default_factory=list,
        description="Main trade-offs between the top candidate products",
    )


class ScoreBreakdown(BaseModel):
    """Transparent mathematical breakdown of an individual product's score."""

    hard_requirement_score: float = Field(
        default=0.0,
        ge=0.0,
        le=40.0,
        description="Points for satisfying non-negotiable requirements (out of 40)",
    )
    budget_score: float = Field(
        default=0.0,
        ge=0.0,
        le=25.0,
        description="Points for budget compliance (out of 25)",
    )
    spec_match_score: float = Field(
        default=0.0,
        ge=0.0,
        le=15.0,
        description="Points for specification match (out of 15)",
    )
    feature_match_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Points for feature matches (out of 10)",
    )
    preference_priority_score: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Points for matching soft preferences and priorities (out of 5)",
    )
    data_confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Points for data completeness and reliability (out of 5)",
    )
    total_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Normalized total score (out of 100.0)",
    )
    violations: List[str] = Field(
        default_factory=list,
        description="Hard constraint violations (e.g. exceeds budget, missing critical hard spec)",
    )


class ProductScoreResult(BaseModel):
    """Product scoring output item."""

    product_name: str = Field(..., description="Product name")
    product: Product = Field(..., description="Verified product object")
    breakdown: ScoreBreakdown = Field(..., description="Detailed score breakdown")
    is_hard_criteria_satisfied: bool = Field(
        default=True,
        description="True if all hard requirements and budget conditions were met",
    )


class DecisionStatus(str, Enum):
    """Outcome classification of the decision process."""

    RECOMMENDED = "recommended"
    NO_SATISFYING_PRODUCT = "no_satisfying_product"
    INSUFFICIENT_DATA = "insufficient_data"


class DecisionResult(BaseModel):
    """Structured decision output produced by the Decision Agent."""

    recommended_product: Optional[Product] = Field(
        default=None,
        description="Top recommended product meeting criteria, or None if no product qualifies",
    )
    alternative_product: Optional[Product] = Field(
        default=None,
        description="Runner-up alternative or closest match with documented trade-offs",
    )
    recommendation_reason: str = Field(
        default="",
        description="Grounded explanation detailing why the product was selected",
    )
    key_advantages: List[str] = Field(
        default_factory=list,
        description="Primary advantages grounded in verified product data",
    )
    tradeoffs: List[str] = Field(
        default_factory=list,
        description="Honest trade-offs, compromises, or downsides",
    )
    unmet_requirements: List[str] = Field(
        default_factory=list,
        description="Any user requirements that could not be satisfied",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the decision based on data completeness and score margin",
    )
    decision_status: DecisionStatus = Field(
        default=DecisionStatus.RECOMMENDED,
        description="Outcome status of the decision",
    )


class FinalDecision(BaseModel):
    """Complete, machine-readable final decision payload."""

    user_requirements: RequirementAnalysisOutput = Field(
        ...,
        description="Original structured user shopping requirements",
    )
    recommended_product: Optional[Product] = Field(
        default=None,
        description="Top recommended product",
    )
    alternative_product: Optional[Product] = Field(
        default=None,
        description="Closest valid alternative",
    )
    comparison_summary: str = Field(
        default="",
        description="Brief summary of candidate comparison",
    )
    score_breakdown: Optional[ScoreBreakdown] = Field(
        default=None,
        description="Scoring breakdown of the recommended product",
    )
    key_reasons: List[str] = Field(
        default_factory=list,
        description="Key supporting reasons for the choice",
    )
    tradeoffs: List[str] = Field(
        default_factory=list,
        description="Noted trade-offs or limitations",
    )
    unmet_requirements: List[str] = Field(
        default_factory=list,
        description="List of unmet criteria",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source citations or URLs for verified product data",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall decision confidence",
    )
    decision_status: DecisionStatus = Field(
        default=DecisionStatus.RECOMMENDED,
        description="Final decision status",
    )


# =====================================================================
# PHASE 5: INTEGRATION SCHEMAS (n8n, SENDGRID, PUSHOVER)
# =====================================================================

class IntegrationDispatchStatus(str, Enum):
    """Dispatch status for an external service."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ServiceDispatchResult(BaseModel):
    """Execution report for a single external integration service."""

    service_name: str = Field(..., description="Service identifier: 'n8n', 'sendgrid', or 'pushover'")
    status: IntegrationDispatchStatus = Field(..., description="'success', 'failed', or 'skipped'")
    message: str = Field(default="", description="Human-readable outcome or failure message")
    timestamp: str = Field(..., description="ISO 8601 timestamp of dispatch attempt")


class IntegrationDispatchSummary(BaseModel):
    """Aggregated report of external automation and notification dispatches."""

    n8n: ServiceDispatchResult = Field(..., description="n8n webhook dispatch result")
    sendgrid: ServiceDispatchResult = Field(..., description="SendGrid email dispatch result")
    pushover: ServiceDispatchResult = Field(..., description="Pushover notification dispatch result")
    any_failures: bool = Field(default=False, description="True if any enabled integration encountered a failure")
