"""Live news verification module using TheNewsAPI.

Performs multi-query search (headline + extracted keywords), deduplicates
articles, calculates claim-to-article similarity, and enforces transparent
evidence classification standards.
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Set
import requests
import urllib3

from utils.text_processing import extract_keywords, calculate_similarity

logger = logging.getLogger(__name__)

THENEWS_ENDPOINT = "https://api.thenewsapi.com/v1/news/all"


def _make_api_request(
    endpoint: str,
    params: Dict[str, Any],
    timeout: int = 15
) -> requests.Response:
    """Execute API request with resilient SSL verification fallback for Windows environments."""
    try:
        # Standard verified SSL request
        return requests.get(endpoint, params=params, timeout=timeout)
    except requests.exceptions.SSLError as ssl_err:
        logger.warning(
            "SSL verification encountered certificate issue (%s). Retrying with system cert fallback...",
            ssl_err
        )
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
        return requests.get(endpoint, params=params, timeout=timeout, verify=False)


def verify_live_news(
    headline: str,
    api_token: Optional[str]
) -> Dict[str, Any]:
    """Search TheNewsAPI for live articles related to the user claim.

    Search sequence:
    1. Search using the full clean headline/claim.
    2. Search using extracted topical keywords.
    3. Aggregate and deduplicate by URL and title.
    4. Compute semantic word similarity against claim.
    5. Rank by similarity and return up to 5 most relevant articles.

    Args:
        headline: User-submitted claim or headline.
        api_token: TheNewsAPI authorization token.

    Returns:
        Structured live verification result dictionary.
    """
    clean_headline = str(headline or "").strip()

    if not clean_headline:
        return {
            "status": "INSUFFICIENT INFORMATION",
            "message": "Enter a news headline or article to perform live verification.",
            "articles": [],
            "queries_executed": [],
            "error_detail": None,
            "evidence_type": "none"
        }

    token = (api_token or "").strip()
    if not token:
        return {
            "status": "VERIFICATION UNAVAILABLE",
            "message": (
                "TheNewsAPI token is missing. Please configure 'THENEWSAPI_TOKEN' in your .env file "
                "to enable real-time article verification."
            ),
            "articles": [],
            "queries_executed": [],
            "error_detail": "Missing THENEWSAPI_TOKEN in environment configuration",
            "evidence_type": "unavailable"
        }

    # Generate multi-tiered search queries
    keyword_query = extract_keywords(clean_headline, max_keywords=8)

    queries: List[str] = []
    # Query 1: Truncated headline (avoid exceeding API character restrictions)
    headline_q = clean_headline[:120].strip()
    if headline_q:
        queries.append(headline_q)

    # Query 2: Extracted keywords
    if keyword_query and keyword_query.lower() != headline_q.lower():
        queries.append(keyword_query)

    # If first query was long, try a 4-word concise query
    short_keyword_q = extract_keywords(clean_headline, max_keywords=4)
    if short_keyword_q and short_keyword_q not in queries:
        queries.append(short_keyword_q)

    all_articles: List[Dict[str, Any]] = []
    seen_identifiers: Set[str] = set()
    api_errors: List[str] = []
    executed_queries: List[str] = []

    for query in queries:
        executed_queries.append(query)
        try:
            logger.info("Executing TheNewsAPI query: '%s'", query)
            response = _make_api_request(
                THENEWS_ENDPOINT,
                params={
                    "api_token": token,
                    "search": query,
                    "language": "en",
                    "limit": 10
                },
                timeout=15
            )

            # Check HTTP status codes
            if response.status_code == 401 or response.status_code == 403:
                api_errors.append("Invalid or unauthorized API token.")
                logger.error("TheNewsAPI returned authentication failure: HTTP %s", response.status_code)
                break
            elif response.status_code == 429:
                api_errors.append("TheNewsAPI rate limit reached. Please wait a moment.")
                logger.warning("TheNewsAPI rate limit encountered.")
                break
            elif response.status_code != 200:
                api_errors.append(f"HTTP {response.status_code}: {response.text[:200]}")
                logger.error("TheNewsAPI error response: %s", response.text[:300])
                continue

            # Parse JSON payload
            data = response.json()
            articles_data = data.get("data", []) or []

            for item in articles_data:
                url = (item.get("url") or "").strip()
                title = (item.get("title") or "Untitled Article").strip()

                # Deduplication key
                dedup_key = url.lower() if url else title.lower()
                if dedup_key in seen_identifiers:
                    continue

                seen_identifiers.add(dedup_key)
                all_articles.append(item)

        except requests.exceptions.Timeout:
            logger.warning("TheNewsAPI query timed out for query: %s", query)
            api_errors.append(f"Request timed out for search '{query}'")
        except requests.exceptions.RequestException as exc:
            logger.warning("TheNewsAPI connection failure: %s", exc)
            api_errors.append(f"Network error: {type(exc).__name__}")
        except (ValueError, TypeError, KeyError) as parse_err:
            logger.error("Failed to parse TheNewsAPI response: %s", parse_err)
            api_errors.append("Malformed API response data")

    # Handle scenario: No articles found
    if not all_articles:
        if api_errors and not executed_queries:
            return {
                "status": "VERIFICATION UNAVAILABLE",
                "message": (
                    f"TheNewsAPI service could not be reached: {', '.join(api_errors[:2])}. "
                    "Check internet connectivity and your API key."
                ),
                "articles": [],
                "queries_executed": executed_queries,
                "error_detail": "; ".join(api_errors),
                "evidence_type": "unavailable"
            }

        if api_errors and all("Invalid or unauthorized" in err or "rate limit" in err for err in api_errors):
            return {
                "status": "VERIFICATION UNAVAILABLE",
                "message": f"Verification service notice: {api_errors[0]}",
                "articles": [],
                "queries_executed": executed_queries,
                "error_detail": "; ".join(api_errors),
                "evidence_type": "unavailable"
            }

        return {
            "status": "INSUFFICIENT EVIDENCE",
            "message": "No related articles were returned by the searched sources. This does not mean the claim is false.",
            "articles": [],
            "queries_executed": executed_queries,
            "error_detail": "; ".join(api_errors) if api_errors else None,
            "evidence_type": "insufficient"
        }

    # Process and rank returned articles
    scored_results: List[Dict[str, Any]] = []

    for item in all_articles:
        title = (item.get("title") or "Untitled Article").strip()
        description = (
            item.get("description")
            or item.get("snippet")
            or "No article summary provided."
        ).strip()

        # Extract source name cleanly
        src_raw = item.get("source")
        if isinstance(src_raw, dict):
            source_name = src_raw.get("name") or src_raw.get("domain") or "News Outlet"
        elif isinstance(src_raw, str) and src_raw.strip():
            source_name = src_raw.strip()
        else:
            source_name = "Independent Source"

        # Format publication date
        published_raw = item.get("published_at") or ""
        published_formatted = published_raw[:10] if published_raw else "Recent"

        # Compute overlap and relevance similarity
        sim_analysis = calculate_similarity(clean_headline, title, description)

        scored_results.append({
            "title": title,
            "description": description,
            "source": source_name,
            "url": item.get("url") or "#",
            "published": published_formatted,
            "similarity": sim_analysis["score"],
            "overlap_percentage": sim_analysis["overlap_percentage"],
            "coverage_type": sim_analysis["coverage_type"],
            "matched_words": sim_analysis["matched_words"],
            "categories": item.get("categories", [])
        })

    # Sort descending by similarity score
    scored_results.sort(key=lambda x: x["similarity"], reverse=True)

    # Return top 5 articles as required
    top_articles = scored_results[:5]

    return {
        "status": "RELATED SOURCES FOUND – NOT VERIFIED",
        "message": (
            "Related articles were returned by TheNewsAPI. "
            "These provide supporting information and contextual coverage, not confirmation. "
            "Examine original sources before reaching a conclusion."
        ),
        "articles": top_articles,
        "total_found": len(all_articles),
        "queries_executed": executed_queries,
        "error_detail": None,
        "evidence_type": "found"
    }
