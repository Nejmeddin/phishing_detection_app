"""Real-time URL analysis view.

Takes a URL from the user, runs it through feature extraction and the LightGBM
model, then explains the verdict rather than just stating it: which factors
drove the score, what was measured, and what the user should do next.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tldextract

from src.config import FEATURE_DISPLAY_NAMES, MODEL_PATH
from src.model.model_loader import ModelLoader
from src.preprocessing.feature_extractor import EnhancedFeatureExtractor

logger = logging.getLogger(__name__)

# Features shown as Yes/No rather than as a raw count.
BOOLEAN_FEATURES = frozenset(
    {
        "IsHTTPS",
        "HasTitle",
        "HasMeta",
        "HasFavicon",
        "HasCopyright",
        "HasSocialNetworking",
        "HasPasswordField",
        "HasSubmitButton",
        "HasKeywordCrypto",
        "HasSuspiciousKeyword",
        "IsTinyURL",
        "DomainAge",
    }
)

# Plain-language explanations surfaced next to the verdict. Each entry pairs a
# predicate over the extracted features with the message to show when it holds.
EXPLANATION_RULES = [
    (
        lambda f: f.get("HasPasswordField") == 1 and f.get("IsHTTPS") == 0,
        "❌ **Unsecured password form**: the site asks for a password without HTTPS.",
    ),
    (
        lambda f: f.get("IsHTTPS") == 1,
        "✅ **Secure connection**: the site uses HTTPS to encrypt data in transit.",
    ),
    (
        lambda f: f.get("HasTitle") == 0,
        "❌ **No page title**: unusual for a legitimate site.",
    ),
    (
        lambda f: f.get("HasFavicon") == 0,
        "❌ **No favicon**: frequently missing on hastily built phishing pages.",
    ),
    (
        lambda f: f.get("NoOfObfuscatedChar", 0) > 0,
        "❌ **Obfuscated URL**: encoded characters may mask the true destination.",
    ),
    (
        lambda f: f.get("HasCopyright") == 1,
        "✅ **Copyright notice**: a trust marker usually present on real sites.",
    ),
    (
        lambda f: f.get("HasSocialNetworking") == 1,
        "✅ **Social network links**: rare on single-purpose phishing pages.",
    ),
    (
        lambda f: f.get("NoOfiFrame", 0) > 2,
        "❌ **Numerous iframes**: often used to embed hidden malicious content.",
    ),
    (
        lambda f: f.get("IsTinyURL") == 1,
        "❌ **Shortened URL**: a link shortener hides the real destination.",
    ),
    (
        lambda f: f.get("HasSuspiciousKeyword") == 1,
        "❌ **Suspicious keyword**: the domain contains trust-baiting vocabulary.",
    ),
]

# Well-known phishing signals, rendered as a reference table.
RISK_INDICATORS = [
    ("Domain registered < 6 months ago", "High", lambda f: f.get("DomainAge", 1) == 1),
    ("Shortened URL", "Medium", lambda f: f.get("IsTinyURL", 0) == 1),
    ("Multiple subdomains", "Medium", lambda f: f.get("NoOfSubDomain", 0) > 1),
    ("Suspicious keywords", "Medium", lambda f: f.get("HasSuspiciousKeyword", 0) == 1),
    ("Obfuscated characters", "High", lambda f: f.get("NoOfObfuscatedChar", 0) > 0),
    ("Served over HTTP", "Medium", lambda f: f.get("IsHTTPS", 1) == 0),
    (
        "Password field without HTTPS",
        "Very High",
        lambda f: f.get("HasPasswordField", 0) == 1 and f.get("IsHTTPS", 0) == 0,
    ),
]


@st.cache_resource
def load_model() -> ModelLoader:
    """Load the LightGBM bundle once per session."""
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Train or restore it first.")
        st.stop()

    loader = ModelLoader(MODEL_PATH)
    if not loader.load():
        st.error("The model file could not be read. Check the pickle format.")
        st.stop()

    return loader


def show_prediction() -> None:
    """Render the real-time prediction page."""
    st.markdown(
        "<h1 class='main-title'>Real-Time URL Analysis</h1>", unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class='info-box'>
        <p>Enter a URL to check whether it is likely legitimate or a phishing
        attempt. The analysis inspects the URL structure, the domain, and the
        content of the page it points to.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("url_form"):
        url_input = st.text_input(
            "URL to analyse:",
            placeholder="https://www.example.com",
            help="Provide the complete URL for the most accurate analysis.",
        )
        analyze_button = st.form_submit_button("Analyse URL")

    if analyze_button and not url_input:
        st.warning("Please enter a URL to analyse.")
    elif analyze_button:
        _run_analysis(url_input)

    with st.expander("How does this analysis work?"):
        st.markdown(
            """
            #### Analysis pipeline

            1. **URL structure** — length, character mix, subdomain depth and
               encoding tricks are measured directly from the string.
            2. **Domain reputation** — the registration age is looked up, since
               newly created domains are disproportionately malicious.
            3. **Page content** — the HTML is fetched and inspected for the
               markers of a credential-harvesting page.
            4. **Classification** — a LightGBM model trained on ~101,000 labelled
               URLs scores the resulting 23-feature vector.
            5. **Explanation** — the factors that drove the score are surfaced so
               you can judge the verdict yourself.
            """
        )

    st.markdown(
        """
        <div class='warning-box'>
        <h3>⚠️ Security reminder</h3>
        <p>This is a decision-support tool, not a guarantee. Stay cautious even
        when a URL is reported as legitimate, especially before entering
        sensitive information.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_analysis(url: str) -> None:
    """Extract features for ``url``, score them, and render the outcome."""
    model_loader = load_model()
    required_features = model_loader.get_required_features()

    if not required_features:
        st.error("Could not determine the features required by the model.")
        return

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    try:
        status_text.text("Extracting features from the URL…")
        extractor = EnhancedFeatureExtractor(required_features)
        features = extractor.extract_features(url)
        progress_bar.progress(0.6)

        status_text.text("Scoring with the model…")
        features_df = model_loader.ensure_feature_consistency(pd.DataFrame([features]))
        probabilities, predictions = model_loader.predict(features_df)
        progress_bar.progress(1.0)

    except (ValueError, TypeError, KeyError) as exc:
        logger.exception("URL analysis failed for %s", url)
        st.error(f"Could not analyse this URL: {exc}")
        return

    finally:
        progress_bar.empty()
        status_text.empty()

    show_prediction_results(url, probabilities, predictions, features, model_loader)


def show_prediction_results(
    url: str,
    probas: np.ndarray,
    predictions: np.ndarray,
    features: dict[str, Any],
    model_loader: ModelLoader,
) -> None:
    """Render the full result report for one analysed URL.

    Args:
        url: The URL that was analysed.
        probas: Phishing probabilities returned by the model.
        predictions: Predicted labels, where 1 means phishing.
        features: The extracted feature mapping.
        model_loader: The loader, used for feature importances.
    """
    is_phishing = bool(predictions[0] == 1)
    phishing_proba = _sanitise_probability(float(np.ravel(probas)[0]), features)
    legitimacy_score = (1.0 - phishing_proba) * 100.0

    logger.info("URL %s scored %.4f (phishing=%s)", url, phishing_proba, is_phishing)

    st.markdown("---")
    st.markdown(
        "<h2 class='section-title'>Analysis Results</h2>", unsafe_allow_html=True
    )

    verdict_col, detail_col = st.columns([1, 2])
    with verdict_col:
        _render_verdict(is_phishing, legitimacy_score)
    with detail_col:
        st.markdown("<h3>Analysed URL</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='word-break: break-all; font-size: 1.05rem;'>{url}</p>",
            unsafe_allow_html=True,
        )
        _render_gauge(legitimacy_score)

    st.markdown("---")
    factors_col, features_col = st.columns(2)
    with factors_col:
        _render_key_factors(features, model_loader)
    with features_col:
        _render_feature_table(features)

    st.markdown("---")
    _render_recommendations(is_phishing)
    _render_technical_details(url, features)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _sanitise_probability(value: float, features: dict[str, Any]) -> float:
    """Return a displayable probability, repairing degenerate model output.

    A NaN score means preprocessing produced something the booster could not
    use. Rather than showing a broken gauge, fall back to a coarse heuristic.
    Exact 0 and 1 are nudged inward so the UI never claims total certainty.
    """
    if np.isnan(value) or np.isinf(value):
        logger.error("Model returned %s; falling back to a heuristic score", value)
        looks_suspicious = (
            features.get("HasPasswordField", 0) == 1
            and features.get("HasSubmitButton", 0) == 1
            and features.get("IsHTTPS", 0) == 0
        )
        return 0.8 if looks_suspicious else 0.2

    return float(np.clip(value, 0.01, 0.99))


def _render_verdict(is_phishing: bool, legitimacy_score: float) -> None:
    """Render the headline verdict card."""
    if is_phishing:
        st.markdown(
            f"""
            <div style="background-color:#4c0519;border-radius:10px;padding:20px;text-align:center;">
            <h1 style="color:#fda4af;margin:0;">⚠️ WARNING</h1>
            <h2 style="color:#fda4af;margin:10px 0;">Potentially malicious site</h2>
            <p style="font-size:1.2rem;color:#fecdd3;">Phishing probability:
            <b>{100 - legitimacy_score:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#052e16;border-radius:10px;padding:20px;text-align:center;">
            <h1 style="color:#86efac;margin:0;">✅ LIKELY SAFE</h1>
            <h2 style="color:#86efac;margin:10px 0;">Probably legitimate site</h2>
            <p style="font-size:1.2rem;color:#bbf7d0;">Legitimacy probability:
            <b>{legitimacy_score:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_gauge(legitimacy_score: float) -> None:
    """Render the legitimacy score as a Plotly gauge."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=legitimacy_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Legitimacy score", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1e3a8a"},
                "steps": [
                    {"range": [0, 30], "color": "#ef4444"},
                    {"range": [30, 70], "color": "#f59e0b"},
                    {"range": [70, 100], "color": "#22c55e"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 4},
                    "thickness": 0.75,
                    "value": legitimacy_score,
                },
            },
        )
    )
    figure.update_layout(height=250, margin={"l": 20, "r": 20, "t": 50, "b": 20})
    st.plotly_chart(figure, use_container_width=True)


def _render_key_factors(features: dict[str, Any], model_loader: ModelLoader) -> None:
    """Show global feature importances plus URL-specific explanations."""
    st.markdown("<h3>Key factors</h3>", unsafe_allow_html=True)

    names, importances = model_loader.get_feature_importance()
    if names and importances:
        ranked = sorted(
            zip(names, importances, strict=False),
            key=lambda pair: pair[1],
            reverse=True,
        )
        top = ranked[:5]
        st.table(
            pd.DataFrame(
                {
                    "Feature": [
                        FEATURE_DISPLAY_NAMES.get(name, name) for name, _ in top
                    ],
                    "Importance": [round(float(score), 2) for _, score in top],
                }
            )
        )

    st.markdown("<h4>What stood out for this URL</h4>", unsafe_allow_html=True)
    explanations = _collect_explanations(features)
    if explanations:
        for explanation in explanations:
            st.markdown(explanation)
    else:
        st.markdown("No distinctive factors were identified for this URL.")


def _collect_explanations(features: dict[str, Any]) -> list[str]:
    """Return every explanation whose condition holds for ``features``."""
    return [message for predicate, message in EXPLANATION_RULES if predicate(features)]


def _render_feature_table(features: dict[str, Any]) -> None:
    """Show every extracted feature in human-readable form."""
    st.markdown("<h3>Extracted features</h3>", unsafe_allow_html=True)

    rows = []
    for name in sorted(features):
        if name == "url":
            continue
        value = features[name]
        if name in BOOLEAN_FEATURES:
            value = "Yes" if value == 1 else "No"
        rows.append((FEATURE_DISPLAY_NAMES.get(name, name), value))

    st.dataframe(
        pd.DataFrame(rows, columns=["Feature", "Value"]),
        use_container_width=True,
        hide_index=True,
    )


def _render_recommendations(is_phishing: bool) -> None:
    """Show the action list matching the verdict."""
    st.markdown("<h3>Security recommendations</h3>", unsafe_allow_html=True)

    if is_phishing:
        st.markdown(
            """
            <div class='warning-box'>
            <h3>🛑 This URL shows signs of phishing</h3>
            <ul>
                <li>Do not visit the site, or leave it immediately</li>
                <li>Never enter credentials or personal information</li>
                <li>Report the URL to your security team or provider</li>
                <li>If you already submitted data, change those passwords now</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class='success-box'>
            <h3>✅ This URL looks legitimate — stay careful anyway</h3>
            <ul>
                <li>Confirm the domain before entering sensitive information</li>
                <li>Enable two-factor authentication wherever it is offered</li>
                <li>Treat unusual requests with suspicion, even on known sites</li>
                <li>Keep your browser and operating system up to date</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_technical_details(url: str, features: dict[str, Any]) -> None:
    """Show the parsed URL and a phishing-indicator checklist."""
    with st.expander("Technical details"):
        parsed = urllib.parse.urlparse(url)
        extracted = tldextract.extract(url)

        st.markdown("#### URL components")
        components = {
            "Scheme": parsed.scheme,
            "Domain": parsed.netloc,
            "Path": parsed.path,
            "Query": parsed.query,
            "Fragment": parsed.fragment,
            "Subdomain": extracted.subdomain,
            "Root domain": extracted.domain,
            "TLD": extracted.suffix,
        }
        st.table(pd.DataFrame(list(components.items()), columns=["Component", "Value"]))

        st.markdown("#### Common phishing indicators")
        st.table(
            pd.DataFrame(
                [
                    {
                        "Indicator": label,
                        "Risk level": level,
                        "Present": "Yes" if predicate(features) else "No",
                    }
                    for label, level, predicate in RISK_INDICATORS
                ]
            )
        )
