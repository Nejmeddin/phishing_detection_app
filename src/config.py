"""Centralised configuration for the phishing detection application.

All tunable parameters, filesystem paths and third-party credentials are
declared here so that the rest of the codebase never hardcodes them.

Secrets are read exclusively from the environment. Copy ``.env.example`` to
``.env`` and fill in your own keys; the file is git-ignored and never leaves
your machine.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional: the app still runs with plain env vars.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - exercised only without the extra.
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PHISHING_DATASET_PATH = RAW_DATA_DIR / "Phishing_Legitimate_full.csv"
MODEL_PATH = PROCESSED_DATA_DIR / "lightgbm_phishing_model.pkl"
MODEL_METRICS_PATH = PROCESSED_DATA_DIR / "model_metrics.json"


# --------------------------------------------------------------------------- #
# Third-party API credentials (optional feature-enrichment services)
# --------------------------------------------------------------------------- #
# These are optional. When a key is absent the corresponding enrichment step is
# skipped and the extractor falls back to locally computed values.
VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "")
URLSCAN_API_KEY = os.environ.get("URLSCAN_API_KEY", "")
WHOISXML_API_KEY = os.environ.get("WHOISXML_API_KEY", "")

VIRUSTOTAL_BASE_URL = "https://www.virustotal.com/api/v3"
URLSCAN_BASE_URL = "https://urlscan.io/api/v1"
WHOISXML_BASE_URL = "https://www.whoisxmlapi.com/whoisserver/WhoisService"

# Shared HTTP settings for every outbound request made by the app.
REQUEST_TIMEOUT = 10  # seconds
REQUEST_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
MODEL_VERSION = "1.0.0"
PREDICTION_THRESHOLD = 0.5  # Probability above which a URL is flagged as phishing.

# The 23 features the trained LightGBM booster expects, in order.
EXPECTED_FEATURES = [
    "IsHTTPS",
    "URLLength",
    "NoOfSubDomain",
    "NoOfDots",
    "NoOfObfuscatedChar",
    "NoOfQmark",
    "NoOfDigits",
    "LineLength",
    "HasTitle",
    "HasMeta",
    "HasFavicon",
    "HasCopyright",
    "HasSocialNetworking",
    "HasPasswordField",
    "HasSubmitButton",
    "HasKeywordCrypto",
    "NoOfPopup",
    "NoOfiFrame",
    "NoOfImage",
    "NoOfJS",
    "NoOfCSS",
    "NoOfURLRedirect",
    "NoOfHyperlink",
]

# Neutral fallback used whenever a feature cannot be computed or fetched.
DEFAULT_FEATURE_VALUES = dict.fromkeys(EXPECTED_FEATURES, 0)

# Features that can only be derived from the page's HTML body.
HTML_FEATURES = frozenset(
    {
        "LineLength",
        "HasTitle",
        "HasMeta",
        "HasFavicon",
        "HasCopyright",
        "HasSocialNetworking",
        "HasPasswordField",
        "HasSubmitButton",
        "HasKeywordCrypto",
        "NoOfPopup",
        "NoOfiFrame",
        "NoOfImage",
        "NoOfJS",
        "NoOfCSS",
        "NoOfURLRedirect",
        "NoOfHyperlink",
    }
)

# Features computable from the URL string alone, without any network call.
URL_FEATURES = frozenset(
    {
        "URLLength",
        "NoOfSubDomain",
        "NoOfDots",
        "NoOfObfuscatedChar",
        "NoOfQmark",
        "NoOfDigits",
    }
)


# --------------------------------------------------------------------------- #
# Heuristics used during feature extraction
# --------------------------------------------------------------------------- #
URL_FEATURES_CONFIG = {
    "suspicious_keywords": [
        "account",
        "authenticate",
        "banking",
        "confirm",
        "customer",
        "ebayisapi",
        "login",
        "password",
        "paypal",
        "secure",
        "signin",
        "support",
        "update",
        "verification",
        "verify",
        "webscr",
    ],
    "suspicious_tlds": [
        "club",
        "jetzt",
        "live",
        "loan",
        "online",
        "party",
        "pw",
        "racing",
        "site",
        "stream",
        "top",
        "win",
        "work",
        "xyz",
    ],
    "shorteners": [
        "bit.ly",
        "cli.gs",
        "goo.gl",
        "is.gd",
        "lnkd.in",
        "ow.ly",
        "short.to",
        "snipurl.com",
        "t.co",
        "tinyurl.com",
        "tr.im",
        "twitthis.com",
        "u.to",
    ],
}

SOCIAL_NETWORK_DOMAINS = [
    "facebook",
    "instagram",
    "linkedin",
    "pinterest",
    "twitter",
    "youtube",
]

CRYPTO_KEYWORDS = [
    "bitcoin",
    "blockchain",
    "crypto",
    "ethereum",
    "token",
    "wallet",
]

JS_REDIRECT_PATTERNS = [
    "window.location",
    "document.location",
    ".href",
]


# --------------------------------------------------------------------------- #
# Streamlit interface
# --------------------------------------------------------------------------- #
STREAMLIT_CONFIG = {
    "page_title": "Phishing Detection",
    "page_icon": "🔒",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme_primary_color": "#1E90FF",
    "theme_secondary_color": "#3D85C6",
    "info_box_color": "#E8F4F8",
    "warning_box_color": "#FFEBEE",
    "success_box_color": "#E8F5E9",
}


# --------------------------------------------------------------------------- #
# Human-readable feature metadata (used by the prediction page)
# --------------------------------------------------------------------------- #
FEATURE_DISPLAY_NAMES = {
    "IsHTTPS": "Uses HTTPS",
    "URLLength": "URL length",
    "NoOfSubDomain": "Number of subdomains",
    "NoOfDots": "Number of dots",
    "NoOfObfuscatedChar": "Obfuscated characters",
    "NoOfQmark": "Number of question marks",
    "NoOfDigits": "Number of digits",
    "LineLength": "HTML line count",
    "HasTitle": "Has a <title> tag",
    "HasMeta": "Has <meta> tags",
    "HasFavicon": "Has a favicon",
    "HasCopyright": "Has a copyright notice",
    "HasSocialNetworking": "Links to social networks",
    "HasPasswordField": "Has a password field",
    "HasSubmitButton": "Has a submit button",
    "HasKeywordCrypto": "Mentions cryptocurrency",
    "NoOfPopup": "Number of popups",
    "NoOfiFrame": "Number of iframes",
    "NoOfImage": "Number of images",
    "NoOfJS": "Number of scripts",
    "NoOfCSS": "Number of stylesheets",
    "NoOfURLRedirect": "Number of redirects",
    "NoOfHyperlink": "Number of hyperlinks",
    "IsTinyURL": "Shortened URL",
    "HasSuspiciousKeyword": "Suspicious keyword in domain",
    "DomainAge": "Recently registered domain",
}

FEATURE_EXPLANATIONS = {
    "IsHTTPS": "Whether the site is served over HTTPS. Its absence is a weak but real warning sign.",
    "URLLength": "Total length of the URL. Phishing URLs tend to be longer to hide the real destination.",
    "NoOfSubDomain": "Number of subdomains. Attackers stack subdomains to mimic a trusted brand.",
    "NoOfDots": "Number of dots in the URL, a proxy for subdomain nesting depth.",
    "NoOfObfuscatedChar": "Percent-encoded characters, often used to disguise the true path.",
    "NoOfQmark": "Number of question marks. More than one suggests a crafted query string.",
    "NoOfDigits": "Digits in the URL, common in generated or spoofed domains.",
    "LineLength": "Number of lines in the HTML source. Phishing pages are often minimal clones.",
    "HasTitle": "Presence of a page title. Hastily built phishing pages frequently omit it.",
    "HasMeta": "Presence of meta tags, usually absent on throwaway pages.",
    "HasFavicon": "Presence of a favicon. Legitimate sites almost always ship one.",
    "HasCopyright": "Presence of a copyright notice, a common trust marker on real sites.",
    "HasSocialNetworking": "Links to social networks, rare on single-purpose phishing pages.",
    "HasPasswordField": "A password input. Expected on login pages but a key phishing ingredient.",
    "HasSubmitButton": "A submit button, needed to actually harvest the credentials.",
    "HasKeywordCrypto": "Cryptocurrency vocabulary, heavily over-represented in scam pages.",
    "NoOfPopup": "Popup-triggering scripts, used to pressure the visitor.",
    "NoOfiFrame": "Number of iframes, often used to embed a hidden malicious payload.",
    "NoOfImage": "Number of images. Clone pages typically hotlink very few.",
    "NoOfJS": "Number of scripts. Both extremes are suspicious.",
    "NoOfCSS": "Number of stylesheets. Real sites ship a full design system.",
    "NoOfURLRedirect": "JavaScript redirects, used to bounce victims through several hosts.",
    "NoOfHyperlink": "Number of links. Phishing pages are usually dead ends.",
    "IsTinyURL": "The URL uses a shortening service, which masks the real destination.",
    "HasSuspiciousKeyword": "The domain contains a trust-baiting word such as 'secure' or 'verify'.",
    "DomainAge": "Whether the domain was registered recently. New domains are far riskier.",
}
