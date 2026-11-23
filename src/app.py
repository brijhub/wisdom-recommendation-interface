"""Streamlit UI. Run with: python -m streamlit run src/app.py"""

import requests
import streamlit as st

from config import get_settings


settings = get_settings()
st.set_page_config(page_title="Wisdom Recommendation", page_icon="💡", layout="centered")
st.title("Wisdom Recommendation")
st.caption("Paste an internal conversation to receive evidence-grounded coaching from the SP01 SME knowledge base.")

with st.expander("What to paste"):
    st.write(
        "Use a plain transcript with speaker names when possible. For example: "
        "`Customer: ...` and `Officer: ...`. No conversation is stored by this application."
    )

transcript = st.text_area(
    "Conversation transcript",
    height=320,
    placeholder="Customer: ...\nOfficer: ...",
    max_chars=50_000,
)

if st.button("Get wisdom", type="primary", use_container_width=True):
    if not transcript.strip():
        st.warning("Paste a conversation transcript first.")
    else:
        try:
            with st.spinner("Reviewing the interaction against SME guidance..."):
                response = requests.post(
                    f"{settings.backend_url.rstrip('/')}/recommend",
                    json={"transcript": transcript},
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
            st.markdown(result["recommendation"])
            with st.expander("Retrieved evidence"):
                for source in result["sources"]:
                    location = source["source_file"]
                    if source.get("sheet"):
                        location += f" · {source['sheet']}"
                    if source.get("section"):
                        location += f" · {source['section']}"
                    if source.get("row_start"):
                        location += f" · row {source['row_start']}"
                    if source.get("reviewer"):
                        location += f" · {source['reviewer']}"
                    st.caption(f"[{source['citation']}] {location} · similarity {source['score']:.3f}")
                    st.text(source["excerpt"])

