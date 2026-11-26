"""Streamlit UI. Run with: python -m streamlit run src/app.py"""

import requests
import streamlit as st

from config import get_settings
from rag import extract_speakers


settings = get_settings()
st.set_page_config(page_title="Wisdom Recommendation", page_icon="💡", layout="centered")
st.markdown(
    """
    <style>
        :root {
            --md-primary: #6750a4;
            --md-primary-dark: #4f378b;
            --md-on-primary: #ffffff;
            --md-surface: #ffffff;
            --md-surface-soft: #f7f2fa;
            --md-outline: #cac4d0;
            --md-text: #1d1b20;
            --md-muted: #49454f;
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 0%, rgba(103, 80, 164, 0.13), transparent 32rem),
                var(--md-surface-soft);
            color: var(--md-text);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
            max-width: 860px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }
        html, body, [class*="st-"] { font-size: 16px; }
        .wisdom-hero {
            display: flex;
            align-items: center;
            gap: 1.15rem;
            margin-bottom: 1.4rem;
            padding: 1.6rem 1.7rem;
            border-radius: 24px;
            color: var(--md-on-primary);
            background: linear-gradient(135deg, #6750a4 0%, #4f378b 100%);
            box-shadow: 0 10px 28px rgba(79, 55, 139, 0.24);
        }
        .wisdom-mark {
            display: grid;
            place-items: center;
            flex: 0 0 52px;
            height: 52px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.17);
            font-size: 1.35rem;
            font-weight: 750;
        }
        .wisdom-eyebrow {
            margin-bottom: 0.25rem;
            font-size: 0.875rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.82;
        }
        .wisdom-hero h1 {
            margin: 0;
            color: white;
            font-size: clamp(1.7rem, 5vw, 2.35rem);
            line-height: 1.12;
            letter-spacing: -0.025em;
        }
        .wisdom-hero p {
            margin: 0.45rem 0 0;
            max-width: 620px;
            color: rgba(255, 255, 255, 0.88);
            line-height: 1.5;
        }
        .section-label {
            margin-bottom: 0.8rem;
            color: var(--md-primary-dark);
            font-size: 0.875rem;
            font-weight: 750;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #e7e0ec;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.96);
            box-shadow: 0 5px 18px rgba(29, 27, 32, 0.08);
        }
        .stTextArea textarea {
            min-height: 300px;
            border: 1px solid var(--md-outline);
            border-radius: 14px;
            background: var(--md-surface);
            color: var(--md-text);
            font-size: 1rem;
            line-height: 1.55;
        }
        .stTextArea textarea:focus {
            border-color: var(--md-primary);
            box-shadow: 0 0 0 2px rgba(103, 80, 164, 0.18);
        }
        .stSelectbox [data-baseweb="select"] > div {
            min-height: 3.25rem;
            border-color: var(--md-outline);
            border-radius: 14px;
            background: var(--md-surface);
        }
        .stButton > button[kind="primary"] {
            min-height: 3.25rem;
            border: none;
            border-radius: 999px;
            background: var(--md-primary);
            color: var(--md-on-primary);
            font-size: 1rem;
            font-weight: 700;
            box-shadow: 0 4px 10px rgba(103, 80, 164, 0.28);
            transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
        }
        .stButton > button[kind="primary"]:hover:not(:disabled) {
            background: var(--md-primary-dark);
            box-shadow: 0 7px 16px rgba(79, 55, 139, 0.32);
            transform: translateY(-1px);
        }
        .stButton > button:disabled { box-shadow: none; }
        [data-testid="stAlert"], details { border-radius: 14px; }
        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--md-muted);
            font-size: 0.875rem;
        }
        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"] { padding-top: 1rem; }
            .wisdom-hero { padding: 1.3rem; border-radius: 20px; }
            .wisdom-mark { display: none; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="wisdom-hero">
        <div class="wisdom-mark">W</div>
        <div>
            <div class="wisdom-eyebrow">Interaction coach</div>
            <h1>Wisdom Recommendation</h1>
            <p>Focused, evidence-grounded coaching for the person you select.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        '<div class="section-label">Conversation review</div>',
        unsafe_allow_html=True,
    )
    transcript = st.text_area(
        "Conversation transcript",
        height=320,
        placeholder=(
            "Ranjit: Completed-project list bhej do.\n"
            "Officer: Ji sir, bhej deta hoon."
        ),
        max_chars=50_000,
    )
    speakers = extract_speakers(transcript)
    person = st.selectbox(
        "Person to receive feedback",
        options=[None, *speakers],
        format_func=lambda value: "Select a person" if value is None else value,
        disabled=not speakers,
    )
    if transcript.strip() and not speakers:
        st.info("No speakers detected. Format each turn as `Person: message`.")
    submitted = st.button(
        "Get feedback",
        type="primary",
        use_container_width=True,
        disabled=not transcript.strip() or person is None,
    )

if submitted:
    try:
        with st.spinner("Reviewing the interaction against SME guidance..."):
            response = requests.post(
                f"{settings.backend_url.rstrip('/')}/recommend",
                json={"transcript": transcript, "person": person},
                timeout=(5, 120),
            )
            response.raise_for_status()
            result = response.json()
    except requests.ConnectionError:
        st.error("Cannot reach FastAPI. Start the backend and try again.")
    except requests.Timeout:
        st.error("The request timed out. Please try again.")
    except requests.HTTPError:
        try:
            detail = response.json().get("detail", "The request failed.")
        except ValueError:
            detail = "The backend returned an unreadable error."
        st.error(detail)
    except (requests.RequestException, ValueError):
        st.error("The backend request failed. Check both terminals for details.")
    else:
        st.markdown(
            '<div class="section-label">Coaching feedback</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(result["recommendation"])
        with st.expander("Retrieved evidence"):
            for source in result["sources"]:
                location = source["source_file"]
                if source.get("sheet"):
                    location += f" • {source['sheet']}"
                if source.get("section"):
                    location += f" • {source['section']}"
                if source.get("row_start"):
                    location += f" • row {source['row_start']}"
                if source.get("reviewer"):
                    location += f" • {source['reviewer']}"
                st.caption(
                    f"[{source['citation']}] {location} • "
                    f"similarity {source['score']:.3f}"
                )
                st.text(source["excerpt"])
