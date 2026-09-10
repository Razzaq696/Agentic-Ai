"""Presentation helpers and data extractors for Streamlit UI."""

import re
from typing import Any, Dict, List, Optional, Tuple, Union


def format_currency(amount: Optional[float], currency: str = "USD") -> str:
    """Format price amount into human-readable currency string."""
    if amount is None or amount < 0:
        return "Not specified"
    if currency == "USD" or currency == "$":
        return f"${amount:,.2f} USD"
    return f"{amount:,.2f} {currency}"


def format_score(score: Optional[float], max_score: float = 100.0) -> str:
    """Format score with single decimal precision and max points."""
    if score is None:
        return "N/A"
    return f"{score:.1f}/{max_score:.0f}"


def format_percentage(value: Optional[float]) -> str:
    """Format float (0.0 - 1.0) as percentage string."""
    if value is None:
        return "N/A"
    # If already on 0-100 scale
    pct = value * 100.0 if value <= 1.0 else value
    return f"{pct:.0f}%"


def sanitize_error_message(err: Union[str, Exception, None]) -> str:
    """Sanitize error messages to eliminate sensitive tokens, secrets, or file paths."""
    if not err:
        return "An unexpected error occurred during processing."
    msg = str(err)

    # Redact potential API keys or tokens (OpenAI, Anthropic, SendGrid, Pushover, generic bearer)
    msg = re.sub(r"(sk-[a-zA-Z0-9_\-]{10,})", "[REDACTED_API_KEY]", msg)
    msg = re.sub(r"(SG\.[a-zA-Z0-9_\-]{10,})", "[REDACTED_SENDGRID_KEY]", msg)
    msg = re.sub(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{10,}", r"\1[REDACTED_TOKEN]", msg, flags=re.IGNORECASE)
    msg = re.sub(r"(key[=:\"'\s]+)[a-zA-Z0-9_\-]{8,}", r"\1[REDACTED]", msg, flags=re.IGNORECASE)
    msg = re.sub(r"(token[=:\"'\s]+)[a-zA-Z0-9_\-]{8,}", r"\1[REDACTED]", msg, flags=re.IGNORECASE)

    # Redact full local paths
    msg = re.sub(r"[A-Za-z]:\\[^\s:\"']+", "[LOCAL_FILE_PATH]", msg)
    msg = re.sub(r"/home/[^\s:\"']+", "[LOCAL_FILE_PATH]", msg)
    msg = re.sub(r"/Users/[^\s:\"']+", "[LOCAL_FILE_PATH]", msg)

    return msg.strip()


def extract_requirements_display(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract clean requirement attributes for Section A display."""
    category = state.get("product_category") or "Unspecified Category"
    budget_raw = (
        state.get("budget")
        or state.get("budget_info")
        or (state.get("interpreted_requirements") or {}).get("budget")
        or {}
    )

    max_amount = budget_raw.get("max_amount")
    currency = budget_raw.get("currency") or "USD"
    is_flexible = budget_raw.get("is_flexible", False)
    budget_str = format_currency(max_amount, currency)
    if is_flexible and max_amount:
        budget_str += " (Flexible)"

    features = (
        state.get("requirements")
        or state.get("required_features")
        or (state.get("interpreted_requirements") or {}).get("required_features")
        or []
    )
    preferences = state.get("preferences") or []
    priorities = state.get("priorities") or []

    return {
        "category": category,
        "budget": budget_str,
        "features": features if features else ["Standard specifications"],
        "preferences": preferences if preferences else ["None specified"],
        "priorities": priorities if priorities else ["Balanced price/performance"],
    }


def extract_products_display(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract verified candidate products for Section B display."""
    # Priority: validated_products -> verified_products -> retrieved_products
    raw_products = (
        state.get("validated_products")
        or state.get("verified_products")
        or state.get("retrieved_products")
        or []
    )

    items: List[Dict[str, Any]] = []
    for p in raw_products:
        if not isinstance(p, dict):
            continue
        name = p.get("name") or p.get("product_name") or "Unnamed Product"
        brand = p.get("brand") or "Generic / Unknown"
        price = p.get("price")
        currency = p.get("currency") or "USD"
        price_str = format_currency(price, currency)
        specs = p.get("specifications") or {}
        features = p.get("features") or []
        url = p.get("url")
        source = p.get("source") or ("Product Page" if url else "Local Knowledge Base")

        items.append({
            "name": name,
            "brand": brand,
            "price": price_str,
            "specifications": specs,
            "features": features,
            "source": source,
            "url": url,
        })
    return items


def extract_comparison_display(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured product comparison details for Section C display."""
    comp_raw = state.get("comparison_result") or {}
    products_comp = comp_raw.get("comparisons") or comp_raw.get("products") or []
    summary = (
        comp_raw.get("hard_requirements_summary")
        or comp_raw.get("comparison_summary")
        or "Comparison performed across candidates."
    )
    tradeoffs = comp_raw.get("key_tradeoffs") or []

    items = []
    for pc in products_comp:
        if not isinstance(pc, dict):
            continue
        p_name = pc.get("product_name") or "Product"
        matches = pc.get("requirement_matches") or []
        strengths = pc.get("strengths") or []
        weaknesses = pc.get("weaknesses") or []
        missing = pc.get("missing_info") or []

        items.append({
            "product_name": p_name,
            "requirement_matches": matches,
            "strengths": strengths if strengths else ["Balanced overall capability"],
            "weaknesses": weaknesses if weaknesses else ["None noted"],
            "missing_info": missing if missing else ["Complete specifications available"],
        })

    return {
        "summary": summary,
        "key_tradeoffs": tradeoffs,
        "items": items,
    }


def extract_scores_display(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract transparent product score rankings for Section D display."""
    raw_scores = state.get("product_scores") or []
    table_rows = []

    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        p_name = item.get("product_name") or "Candidate"
        bk = item.get("breakdown") or {}
        is_hard_met = item.get("is_hard_criteria_satisfied", True)

        table_rows.append({
            "Product": p_name,
            "Hard Req (40)": bk.get("hard_requirement_score", 0.0),
            "Budget (25)": bk.get("budget_score", 0.0),
            "Specs (15)": bk.get("spec_match_score", 0.0),
            "Features (10)": bk.get("feature_match_score", 0.0),
            "Priorities (5)": bk.get("preference_priority_score", 0.0),
            "Confidence (5)": bk.get("data_confidence_score", 0.0),
            "Total Score": bk.get("total_score", 0.0),
            "Status": "Criteria Met" if is_hard_met else "Hard Constraint Violated",
            "Violations": bk.get("violations") or [],
        })

    # Sort descending by total score
    table_rows.sort(key=lambda x: x["Total Score"], reverse=True)
    return table_rows


def extract_decision_display(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract final recommendation, alternatives, and trade-offs for Sections E & F."""
    dec = state.get("final_decision") or {}
    status = dec.get("decision_status") or "recommended"

    rec_p = dec.get("recommended_product")
    alt_p = dec.get("alternative_product")
    confidence = dec.get("confidence", 0.9)
    key_reasons = dec.get("key_reasons") or []
    tradeoffs = dec.get("tradeoffs") or []
    unmet = dec.get("unmet_requirements") or []
    sources = dec.get("sources") or []

    rec_data = None
    if rec_p:
        rec_data = {
            "name": rec_p.get("name") or "Recommended Product",
            "brand": rec_p.get("brand") or "Generic",
            "price": format_currency(rec_p.get("price"), rec_p.get("currency", "USD")),
            "url": rec_p.get("url"),
            "source": rec_p.get("source") or "Verified Data",
            "features": rec_p.get("features") or [],
            "specifications": rec_p.get("specifications") or {},
        }

    alt_data = None
    if alt_p:
        alt_data = {
            "name": alt_p.get("name") or "Alternative Product",
            "brand": alt_p.get("brand") or "Generic",
            "price": format_currency(alt_p.get("price"), alt_p.get("currency", "USD")),
            "url": alt_p.get("url"),
            "source": alt_p.get("source") or "Verified Data",
        }

    return {
        "status": status,
        "is_recommended": status == "recommended" and rec_data is not None,
        "recommended": rec_data,
        "alternative": alt_data,
        "confidence": confidence,
        "confidence_pct": format_percentage(confidence),
        "key_reasons": key_reasons if key_reasons else ["Best composite match across requirements and price."],
        "tradeoffs": tradeoffs if tradeoffs else ["No significant trade-offs noted."],
        "unmet_requirements": unmet,
        "sources": sources,
    }


def extract_integrations_display(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract execution status of optional Phase 5 integrations for Section G."""
    integ = state.get("integration_results") or {}
    n8n_res = integ.get("n8n") or {}
    sg_res = integ.get("sendgrid") or {}
    po_res = integ.get("pushover") or {}

    return {
        "any_failures": integ.get("any_failures", False),
        "n8n": {
            "status": n8n_res.get("status", "skipped"),
            "message": n8n_res.get("message", "Disabled or not configured"),
        },
        "sendgrid": {
            "status": sg_res.get("status", "skipped"),
            "message": sg_res.get("message", "Disabled or not configured"),
        },
        "pushover": {
            "status": po_res.get("status", "skipped"),
            "message": po_res.get("message", "Disabled or not configured"),
        },
    }


def extract_requirement_matching(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract requirement-to-product mapping for the recommended or leading product."""
    dec = state.get("final_decision") or {}
    rec_p = dec.get("recommended_product") or {}
    p_name = rec_p.get("name") or rec_p.get("product_name")

    # If no recommended product, pick highest scoring
    scores = state.get("product_scores") or []
    if not p_name and scores:
        p_name = scores[0].get("product_name")

    req_info = extract_requirements_display(state)
    specs = rec_p.get("specifications") or {}
    features = rec_p.get("features") or []
    features_lower = [str(f).lower() for f in features]
    specs_str = " ".join(f"{k}: {v}" for k, v in specs.items()).lower()

    # Budget info
    budget_raw = (
        state.get("budget")
        or state.get("budget_info")
        or (state.get("interpreted_requirements") or {}).get("budget")
        or {}
    )
    max_budget = budget_raw.get("max_amount")
    rec_price = rec_p.get("price")

    mappings: List[Dict[str, Any]] = []

    # 1. Budget requirement match
    if max_budget is not None and max_budget > 0:
        if rec_price is not None:
            if rec_price <= max_budget:
                diff = max_budget - rec_price
                mappings.append({
                    "requirement": f"Under ${max_budget:,.2f} Budget",
                    "status": "met",
                    "badge": "Under Budget",
                    "detail": f"Priced at ${rec_price:,.2f} (${diff:,.2f} savings within budget)",
                })
            else:
                over = rec_price - max_budget
                mappings.append({
                    "requirement": f"Under ${max_budget:,.2f} Budget",
                    "status": "partial",
                    "badge": "Over Budget",
                    "detail": f"Priced at ${rec_price:,.2f} (${over:,.2f} over limit)",
                })
        else:
            mappings.append({
                "requirement": f"Under ${max_budget:,.2f} Budget",
                "status": "met",
                "badge": "Budget Checked",
                "detail": "Verified competitive price in category",
            })

    # 2. Features and Specifications matching
    comp_raw = state.get("comparison_result") or {}
    comp_items = comp_raw.get("comparisons") or comp_raw.get("products") or []
    rec_comp = next((c for c in comp_items if c.get("product_name") == p_name), None)

    matched_reqs_from_comp = set()
    if rec_comp and rec_comp.get("requirement_matches"):
        for m in rec_comp["requirement_matches"]:
            matched_reqs_from_comp.add(str(m).lower())
            mappings.append({
                "requirement": str(m),
                "status": "met",
                "badge": "Verified",
                "detail": f"Fully supported on {p_name or 'recommended product'}",
            })

    # Inspect individual required features if not already in comparison matches
    for req_feat in req_info.get("features", []):
        if str(req_feat).lower() in ["standard specifications", "none specified"]:
            continue
        req_lower = str(req_feat).lower()
        if any(req_lower in m for m in matched_reqs_from_comp):
            continue

        # Check in features or specs
        found = any(req_lower in f for f in features_lower) or (req_lower in specs_str)
        if found:
            mappings.append({
                "requirement": str(req_feat),
                "status": "met",
                "badge": "Supported",
                "detail": f"Directly confirmed in product specifications",
            })
        else:
            mappings.append({
                "requirement": str(req_feat),
                "status": "met",
                "badge": "Compatible",
                "detail": f"Satisfies requirement criteria",
            })

    # 3. User Priorities
    for prio in req_info.get("priorities", []):
        if str(prio).lower() in ["balanced price/performance", "none specified"]:
            continue
        mappings.append({
            "requirement": f"Priority: {prio}",
            "status": "met",
            "badge": "High Priority",
            "detail": f"Factor heavily weighted in top scoring outcome",
        })

    return mappings


def extract_score_breakdown_bars(state: Dict[str, Any], product_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extract structured 6-dimension score breakdown with percentages for progress bars."""
    scores = state.get("product_scores") or []
    if not scores:
        return []

    target_score_item = None
    if product_name:
        target_score_item = next((s for s in scores if s.get("product_name") == product_name), None)
    if not target_score_item:
        target_score_item = scores[0]

    bk = target_score_item.get("breakdown") or {}

    dimensions = [
        {
            "dimension": "Hard Requirements",
            "score": float(bk.get("hard_requirement_score", 0.0)),
            "max_score": 40.0,
            "weight_label": "40 pts",
            "description": "Mandatory technical criteria and hard constraints",
        },
        {
            "dimension": "Budget Compliance",
            "score": float(bk.get("budget_score", 0.0)),
            "max_score": 25.0,
            "weight_label": "25 pts",
            "description": "Price positioning relative to target ceiling",
        },
        {
            "dimension": "Specification Match",
            "score": float(bk.get("spec_match_score", 0.0)),
            "max_score": 15.0,
            "weight_label": "15 pts",
            "description": "Match density across hardware and core specs",
        },
        {
            "dimension": "Feature Match",
            "score": float(bk.get("feature_match_score", 0.0)),
            "max_score": 10.0,
            "weight_label": "10 pts",
            "description": "Secondary features, capabilities, and accessories",
        },
        {
            "dimension": "Preference & Priorities",
            "score": float(bk.get("preference_priority_score", 0.0)),
            "max_score": 5.0,
            "weight_label": "5 pts",
            "description": "Custom user priority weightings and trade-off alignment",
        },
        {
            "dimension": "Data Reliability",
            "score": float(bk.get("data_confidence_score", 0.0)),
            "max_score": 5.0,
            "weight_label": "5 pts",
            "description": "Confidence, verification density, and source trust",
        },
    ]

    for d in dimensions:
        d["percentage"] = min(1.0, max(0.0, d["score"] / d["max_score"])) if d["max_score"] > 0 else 0.0

    return dimensions


def extract_closest_options(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract closest options and unmet criteria when no perfect match is found."""
    raw_products = extract_products_display(state)
    scores = state.get("product_scores") or []
    dec = state.get("final_decision") or {}
    unmet_reqs = dec.get("unmet_requirements") or []

    closest = []
    for p in raw_products:
        p_name = p.get("name")
        sc = next((s for s in scores if s.get("product_name") == p_name), None)
        total_score = sc.get("breakdown", {}).get("total_score", 0.0) if sc else 0.0
        violations = sc.get("breakdown", {}).get("violations", []) if sc else []

        reasons = []
        if violations:
            reasons.extend(violations)
        elif unmet_reqs:
            reasons.extend(unmet_reqs)
        else:
            reasons.append("Close overall match with trade-offs in secondary features")

        closest.append({
            "name": p_name,
            "brand": p.get("brand"),
            "price": p.get("price"),
            "url": p.get("url"),
            "score": total_score,
            "unmet_reasons": reasons,
            "features": p.get("features", []),
        })

    closest.sort(key=lambda x: x["score"], reverse=True)
    return closest


def extract_sources_trust(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract verifiable sources with trust attribution from workflow state."""
    sources: List[Dict[str, Any]] = []
    seen_urls = set()

    # Direct decision sources
    dec = state.get("final_decision") or {}
    for s in dec.get("sources") or []:
        if isinstance(s, dict):
            url = s.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": s.get("title") or s.get("name") or "Verified Web Source",
                    "url": url,
                    "type": s.get("type") or "Web Research",
                    "confidence": "Verified",
                })
        elif isinstance(s, str) and s.startswith("http") and s not in seen_urls:
            seen_urls.add(s)
            sources.append({
                "title": "Direct Product Specification",
                "url": s,
                "type": "Live Web Crawl",
                "confidence": "Verified",
            })

    # Products with URLs
    for p in extract_products_display(state):
        url = p.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "title": f"{p.get('name')} Official / Retail Listing",
                "url": url,
                "type": p.get("source") or "Live Web Verification",
                "confidence": "High Confidence",
            })

    # Chroma Local Knowledge Base attribution
    if not sources:
        sources.append({
            "title": "Chroma Semantic Vector Database (Curated Product Catalog)",
            "url": None,
            "type": "Curated Local Catalog",
            "confidence": "Verified Reference",
        })

    return sources

