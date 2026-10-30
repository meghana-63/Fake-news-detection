"""Text processing and similarity computation utilities for TruthLens AI.

This module provides text cleaning matching the original ML model training
pipeline, keyword extraction for search query formulation, and text overlap
similarity scoring.
"""

import re
import string
from typing import Dict, List, Set, Any


# Standard English stop words for query generation
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "says", "said", "say", "saying", "claim", "claims",
    "reported", "reports", "according", "new", "latest", "just", "also", "one",
    "two", "first", "today", "yesterday", "tomorrow", "breaking", "update"
}


def clean_text(text: str) -> str:
    """Normalize and clean text matching the model training preprocessing."""
    text = str(text or "")
    # Remove news agency datelines if present (e.g. WASHINGTON (Reuters) - )
    text = re.sub(r'^[A-Za-z\s]+(?:\(Reuters\)|\(reuters\)|\(REUTERS\))\s*[-–—]\s*', ' ', text)
    text = re.sub(r'\b(reuters|reuter)\b', ' ', text, flags=re.IGNORECASE)
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+|http\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_keywords(text: str, max_keywords: int = 8) -> str:
    """Extract significant topical keywords for concise API search queries.

    Args:
        text: Raw headline or user query string.
        max_keywords: Maximum number of distinct keywords to return.

    Returns:
        Space-separated keyword query string.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""

    # Strip URLs and formatting
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", raw)
    tokens = re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", cleaned)

    keywords: List[str] = []
    seen: Set[str] = set()

    for token in tokens:
        normalized = token.lower().strip("'-")
        if (
            len(normalized) < 3
            or normalized in STOP_WORDS
            or normalized in seen
        ):
            continue

        keywords.append(token.strip("'-"))
        seen.add(normalized)

        if len(keywords) >= max_keywords:
            break

    if keywords:
        return " ".join(keywords)

    # Fallback: truncate original cleaned text
    words = [w for w in cleaned.split() if w.lower() not in STOP_WORDS]
    return " ".join(words[:max_keywords]) if words else raw[:120]


def calculate_similarity(claim: str, title: str, description: str = "") -> Dict[str, Any]:
    """Calculate word overlap and Jaccard similarity between a claim and an article.

    Args:
        claim: User-submitted headline or statement.
        title: Article title from live news API.
        description: Article summary or snippet.

    Returns:
        Dictionary with:
            - score: float percentage (0.0 to 100.0)
            - overlap_percentage: float percentage
            - jaccard_percentage: float percentage
            - coverage_type: "Supporting coverage" | "Related context" | "Peripheral mention"
            - matched_words: list of common terms
    """
    claim_clean = clean_text(claim)
    title_clean = clean_text(title)
    desc_clean = clean_text(description)

    claim_words = {w for w in claim_clean.split() if len(w) > 2 and w not in STOP_WORDS}
    article_words = {
        w for w in (title_clean + " " + desc_clean).split()
        if len(w) > 2 and w not in STOP_WORDS
    }
    title_words = {w for w in title_clean.split() if len(w) > 2 and w not in STOP_WORDS}

    if not claim_words or not article_words:
        return {
            "score": 0.0,
            "overlap_percentage": 0.0,
            "jaccard_percentage": 0.0,
            "coverage_type": "Insufficient overlap",
            "matched_words": []
        }

    matched = claim_words & article_words
    title_matched = claim_words & title_words

    # Calculate claim coverage (how much of the claim is covered in the article)
    claim_coverage = (len(matched) / len(claim_words)) * 100.0

    # Calculate Jaccard similarity between claim and full article text
    union_len = len(claim_words | article_words)
    jaccard = (len(matched) / union_len) * 100.0 if union_len > 0 else 0.0

    # Title-weighted blend: title matches indicate higher direct relevance
    title_bonus = min(25.0, (len(title_matched) / len(claim_words)) * 25.0) if claim_words else 0.0
    composite_score = min(100.0, (claim_coverage * 0.75) + title_bonus)

    # Classify coverage alignment
    if composite_score >= 60.0:
        coverage_type = "Supporting coverage"
    elif composite_score >= 30.0:
        coverage_type = "Related context"
    else:
        coverage_type = "Peripheral mention"

    return {
        "score": round(composite_score, 1),
        "overlap_percentage": round(claim_coverage, 1),
        "jaccard_percentage": round(jaccard, 1),
        "coverage_type": coverage_type,
        "matched_words": sorted(list(matched))
    }
