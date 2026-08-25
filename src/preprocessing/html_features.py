"""Derive model features from a page's HTML body.

Both the live extractor and the enrichment fallbacks need these computations,
and they must agree: a feature scraped directly has to mean exactly what the
same feature recovered from a cached snapshot means, or the model sees an
inconsistent vector. Keeping the logic here is what guarantees that.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup

from src.config import (
    CRYPTO_KEYWORDS,
    HTML_FEATURES,
    JS_REDIRECT_PATTERNS,
    SOCIAL_NETWORK_DOMAINS,
)

_POPUP_PATTERN = re.compile(r"window\.open|popup|alert")


def extract_html_features(
    html_content: str, wanted: Iterable[str] | None = None
) -> dict[str, int]:
    """Compute HTML-derived features from a page body.

    Args:
        html_content: The raw HTML source of the page.
        wanted: Feature names to compute. Defaults to every HTML feature.

    Returns:
        A mapping of feature name to value, restricted to ``wanted``.
    """
    requested = set(wanted) if wanted is not None else set(HTML_FEATURES)
    requested &= HTML_FEATURES
    if not requested:
        return {}

    soup = BeautifulSoup(html_content, "html.parser")
    lowered = html_content.lower()
    # The rendered text, so that entity-encoded markers such as `&copy;` are
    # seen the same way a visitor would see them.
    rendered = soup.get_text().lower()

    computations = {
        "LineLength": lambda: len(html_content.splitlines()),
        "HasTitle": lambda: 1 if soup.title else 0,
        "HasMeta": lambda: 1 if soup.find_all("meta") else 0,
        "HasFavicon": lambda: _has_favicon(soup),
        "HasCopyright": lambda: (
            1
            if "©" in html_content
            or "©" in rendered
            or "copyright" in lowered
            or "copyright" in rendered
            else 0
        ),
        "HasSocialNetworking": lambda: (
            1 if any(name in lowered for name in SOCIAL_NETWORK_DOMAINS) else 0
        ),
        "HasPasswordField": lambda: (
            1 if soup.find_all("input", {"type": "password"}) else 0
        ),
        "HasSubmitButton": lambda: _has_submit_button(soup),
        "HasKeywordCrypto": lambda: (
            1 if any(keyword in lowered for keyword in CRYPTO_KEYWORDS) else 0
        ),
        "NoOfPopup": lambda: len(soup.find_all("script", string=_POPUP_PATTERN)),
        "NoOfiFrame": lambda: len(soup.find_all("iframe")),
        "NoOfImage": lambda: len(soup.find_all("img")),
        "NoOfJS": lambda: len(soup.find_all("script")),
        "NoOfCSS": lambda: (
            len(soup.find_all("link", {"rel": "stylesheet"}))
            + len(soup.find_all("style"))
        ),
        "NoOfURLRedirect": lambda: _count_js_redirects(soup),
        "NoOfHyperlink": lambda: len(soup.find_all("a")),
    }

    return {name: computations[name]() for name in requested if name in computations}


def _has_favicon(soup: BeautifulSoup) -> int:
    """Return 1 when the document declares an icon link of any kind."""
    for link in soup.find_all("link"):
        rel = link.get("rel", "")
        if isinstance(rel, list):
            rel = " ".join(rel)
        if "icon" in rel.lower():
            return 1
    return 0


def _has_submit_button(soup: BeautifulSoup) -> int:
    """Return 1 when the document contains a submit control of any kind."""
    if soup.find_all("input", {"type": "submit"}):
        return 1
    if soup.find_all("button", {"type": "submit"}):
        return 1
    return 0


def _count_js_redirects(soup: BeautifulSoup) -> int:
    """Count inline scripts that navigate the browser elsewhere.

    Counted per script rather than per pattern match: a single statement such as
    ``document.location.href = ...`` matches several patterns at once but is
    still only one redirect.
    """
    return sum(
        1
        for script in soup.find_all("script")
        if script.string
        and any(pattern in script.string for pattern in JS_REDIRECT_PATTERNS)
    )
