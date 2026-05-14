"""
eThute Lenna — Voice & Language Integration Module
===================================================
Drop this file into your eThute Lenna project root.
Provides:
  • Text-to-Speech (TTS) in isiZulu, Sesotho, and English
  • Translation of study guide / exam paper text via Google Translate
  • A reusable Streamlit sidebar widget to plug into any page
  • A standalone read-aloud player for PDF / text content

Dependencies (add to requirements.txt):
    gtts>=2.3.2
    googletrans==4.0.0rc1
    PyPDF2>=3.0.1
    streamlit>=1.30.0
"""

import io
import base64
import textwrap
import streamlit as st
from gtts import gTTS
from googletrans import Translator

# ─────────────────────────────────────────────
# LANGUAGE CONFIG
# ─────────────────────────────────────────────

LANGUAGES = {
    "English":  {"gtts_lang": "en",  "trans_dest": "en",  "flag": "🇿🇦"},
    "isiZulu":  {"gtts_lang": "zu",  "trans_dest": "zu",  "flag": "🌍"},
    "Sesotho":  {"gtts_lang": "st",  "trans_dest": "st",  "flag": "🌍"},
}

# ─────────────────────────────────────────────
# CORE: TEXT-TO-SPEECH
# ─────────────────────────────────────────────

def text_to_speech(text: str, lang_code: str = "en") -> bytes:
    """Convert text to MP3 audio bytes using gTTS.

    Args:
        text:      The text to speak aloud.
        lang_code: BCP-47 language code ('en', 'zu', or 'st').

    Returns:
        MP3 audio as raw bytes.
    """
    tts = gTTS(text=text, lang=lang_code, slow=False)
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    return mp3_buffer.read()


def audio_html_player(audio_bytes: bytes) -> str:
    """Return an HTML <audio> tag with inline base64-encoded MP3."""
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return (
        f'<audio controls autoplay style="width:100%; margin-top:8px;">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        f"Your browser does not support the audio element."
        f"</audio>"
    )


# ─────────────────────────────────────────────
# CORE: TRANSLATION
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def translate_text(text: str, dest_lang: str) -> str:
    """Translate text into the target language using Google Translate.

    Results are cached so the same chunk is not re-translated on reruns.

    Args:
        text:      Source text (assumed English).
        dest_lang: ISO 639-1 language code ('zu' or 'st').

    Returns:
        Translated string, or the original text on failure.
    """
    if dest_lang == "en":
        return text
    try:
        translator = Translator()
        result = translator.translate(text, dest=dest_lang)
        return result.text
    except Exception as e:
        st.warning(f"⚠️ Translation unavailable: {e}. Showing original text.")
        return text


# ─────────────────────────────────────────────
# STREAMLIT WIDGET: LANGUAGE + VOICE SIDEBAR
# ─────────────────────────────────────────────

def render_language_sidebar() -> dict:
    """Render the language & voice settings panel in the sidebar.

    Call once at the top of each Streamlit page.  Returns the
    currently selected language config dict.

    Example
    -------
    >>> lang = render_language_sidebar()
    >>> displayed_text = get_display_text(raw_english_text, lang)
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🌐 Language / Ulimi / Puo")

        selected_label = st.selectbox(
            "Choose language · Khetha ulimi · Kgetha puo",
            options=list(LANGUAGES.keys()),
            index=0,
        )
        lang = LANGUAGES[selected_label]
        lang["label"] = selected_label

        st.markdown("### 🔊 Voice / Izwi / Lentswe")
        lang["voice_enabled"] = st.toggle("Read aloud · Funda kakhulu · Bala phahameng", value=False)
        lang["voice_speed"] = st.select_slider(
            "Speed · Isivinini · Lebelo",
            options=["Slow", "Normal"],
            value="Normal",
        )
        st.markdown("---")

    return lang


# ─────────────────────────────────────────────
# HELPER: TRANSLATE + DISPLAY
# ─────────────────────────────────────────────

def get_display_text(english_text: str, lang: dict) -> str:
    """Return the text in the selected language (translate if needed)."""
    dest = lang["trans_dest"]
    if dest == "en":
        return english_text
    with st.spinner(f"Translating to {lang['label']}…"):
        return translate_text(english_text, dest)


def render_text_with_voice(
    text: str,
    lang: dict,
    container=None,
    chunk_size: int = 500,
) -> None:
    """Display text and optionally play it as audio.

    For long texts the TTS is split into manageable chunks because
    gTTS has a practical limit of ~500 characters per request on
    the free tier.

    Args:
        text:       Text to display (already in the target language).
        lang:       Language dict from render_language_sidebar().
        container:  Streamlit container (defaults to main area).
        chunk_size: Max characters per TTS chunk.
    """
    target = container if container else st
    target.markdown(text)

    if lang.get("voice_enabled"):
        slow = lang.get("voice_speed") == "Slow"
        chunks = textwrap.wrap(text, chunk_size, break_long_words=False)
        for i, chunk in enumerate(chunks, 1):
            label = f"▶ Play part {i}" if len(chunks) > 1 else "▶ Play audio"
            with target.expander(label, expanded=(i == 1)):
                try:
                    tts = gTTS(text=chunk, lang=lang["gtts_lang"], slow=slow)
                    buf = io.BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    target.audio(buf, format="audio/mp3")
                except Exception as e:
                    target.error(f"TTS error: {e}")


# ─────────────────────────────────────────────
# HELPER: READ A PDF PAGE ALOUD
# ─────────────────────────────────────────────

def extract_pdf_text(pdf_file) -> str:
    """Extract plain text from an uploaded PDF file object."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except Exception as e:
        return f"[Could not extract PDF text: {e}]"


def render_pdf_reader(lang: dict) -> None:
    """Full PDF upload → translate → read-aloud widget.

    Paste this call into any Streamlit page where you want users
    to upload study guides or past exam papers.
    """
    st.markdown("### 📄 Upload Study Guide / Past Exam Paper")
    uploaded = st.file_uploader(
        "Upload a PDF · Layisha i-PDF · Kenya PDF",
        type=["pdf"],
        help="Supports study guides and past exam papers",
    )

    if not uploaded:
        return

    with st.spinner("Reading document…"):
        raw_text = extract_pdf_text(uploaded)

    if not raw_text.strip():
        st.warning("No readable text found in this PDF.")
        return

    st.success(f"✅ Loaded **{uploaded.name}** ({len(raw_text):,} characters)")

    # Page-by-page selector
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    total = len(paragraphs)

    if total > 1:
        page_range = st.slider(
            "Select section · Khetha isigaba · Kgetho karolo",
            min_value=1, max_value=total,
            value=(1, min(5, total)),
        )
        selected = "\n\n".join(paragraphs[page_range[0]-1 : page_range[1]])
    else:
        selected = raw_text

    # Translate
    display_text = get_display_text(selected, lang)

    st.markdown("#### 📖 Content")
    render_text_with_voice(display_text, lang)
