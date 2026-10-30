"""Evidence synthesis and comparison engine for TruthLens AI.

Synthesizes machine learning pattern predictions with live news retrieval
to provide objective, contextual evidence breakdowns without declaring
unsupported factual absolutes.
"""

from typing import Dict, List, Any, Optional


def build_evidence_analysis(
    headline: str,
    prediction_result: Dict[str, Any],
    verification_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate structured comparison table and comprehensive final explanation.

    Args:
        headline: User-submitted news headline or text snippet.
        prediction_result: Output from predict_news().
        verification_result: Output from verify_live_news().

    Returns:
        Structured evidence breakdown dictionary.
    """
    articles = verification_result.get("articles", [])
    v_status = verification_result.get("status", "INSUFFICIENT EVIDENCE")
    pred_label = prediction_result.get("label", "MODEL UNAVAILABLE")
    confidence = prediction_result.get("confidence")

    comparison_items: List[Dict[str, Any]] = []

    # Map each retrieved article into comparison matrix
    for idx, art in enumerate(articles, start=1):
        sim = art.get("similarity", 0.0)
        coverage_type = art.get("coverage_type", "Related context")

        # Determine alignment indicator
        if sim >= 60.0:
            alignment = "High Direct Alignment"
            badge = "badge-high"
        elif sim >= 30.0:
            alignment = "Substantive Context"
            badge = "badge-medium"
        else:
            alignment = "Peripheral Mention"
            badge = "badge-low"

        comparison_items.append({
            "index": idx,
            "title": art.get("title", "Untitled"),
            "source": art.get("source", "Unknown Source"),
            "published": art.get("published", "N/A"),
            "similarity": sim,
            "overlap_percentage": art.get("overlap_percentage", 0.0),
            "alignment": alignment,
            "badge_class": badge,
            "url": art.get("url", "#"),
            "description": art.get("description", "")
        })

    # Synthesize Final Objective Explanation
    explanation_parts: List[str] = []

    # 1. Model assessment component
    if pred_label == "REAL NEWS":
        conf_str = f" with {confidence}% confidence" if confidence is not None else ""
        explanation_parts.append(
            f"The NLP classification model evaluated the linguistic structure, vocabulary distribution, "
            f"and stylistic markers of the text and associated it with patterns typical of credible reporting{conf_str}."
        )
    elif pred_label == "FAKE NEWS":
        conf_str = f" with {confidence}% confidence" if confidence is not None else ""
        explanation_parts.append(
            f"The NLP classification model identified lexical and stylistic patterns frequently observed in sensationalized "
            f"or fabricated content{conf_str}."
        )
    else:
        explanation_parts.append(
            "The statistical machine learning classifier was unable to generate a pattern prediction for this text."
        )

    # 2. Live verification component
    if v_status == "RELATED SOURCES FOUND – NOT VERIFIED":
        count = len(articles)
        top_src = articles[0].get("source", "major news sources") if articles else "established outlets"
        explanation_parts.append(
            f"Live cross-referencing via TheNewsAPI located {count} related news article{'s' if count > 1 else ''} "
            f"from independent outlets including {top_src}. This demonstrates contemporary reporting on this subject matter."
        )
        explanation_parts.append(
            "While the presence of related coverage provides valuable context, users should review the linked articles "
            "directly to confirm details, quotations, and official statements."
        )
    elif v_status == "INSUFFICIENT EVIDENCE":
        explanation_parts.append(
            "The real-time news search yielded no matching articles across indexed publishers. "
            "This lack of indexed coverage may indicate an obscure topic, hyper-local event, or recent development, "
            "and should NOT be interpreted as conclusive proof that the claim is false."
        )
    else:
        explanation_parts.append(
            "Live source verification was currently unavailable due to an API service condition. "
            "Cross-referencing could not be performed against live news feeds."
        )

    # Overall synthesis conclusion
    explanation_summary = " ".join(explanation_parts)

    # Dynamic Verification Status
    if "RELATED" in v_status:
        if "REAL" in pred_label:
            v_badge = "badge-real"
            v_status_label = "Corroborated by News Index"
        else:
            v_badge = "badge-found"
            v_status_label = "Related Sources Located"
    elif "INSUFFICIENT" in v_status:
        v_badge = "badge-insufficient"
        v_status_label = "Insufficient Live Evidence"
    else:
        v_badge = "badge-unavailable"
        v_status_label = "Verification Offline"

    return {
        "headline": headline,
        "comparison_items": comparison_items,
        "articles_count": len(comparison_items),
        "status": v_status,
        "verification_status_label": v_status_label,
        "verification_status_badge": v_badge,
        "explanation": explanation_summary,
        "limitations": (
            "Machine-learning classifications reflect statistical correlations in historical training datasets, "
            "and search results reflect published articles in news indexes. Neither metric represents an absolute "
            "guarantee of factual accuracy. Always verify critical claims with primary documentation."
        )
    }
