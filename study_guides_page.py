"""
eThute Lenna — Study Guides Page (with Voice & Language)
=========================================================
Replace or merge this with your existing study_guides page.

How to run (from project root):
    streamlit run study_guides_page.py
"""

import streamlit as st
from language_voice import (
    render_language_sidebar,
    render_pdf_reader,
    get_display_text,
    render_text_with_voice,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="eThute Lenna — Study Guides",
    page_icon="📚",
    layout="wide",
)

# ── Language / voice selector (renders in sidebar) ───────────────────────────
lang = render_language_sidebar()

# ── Page header (translated) ──────────────────────────────────────────────────
TITLES = {
    "English": "📚 Study Guides & Past Exam Papers",
    "isiZulu": "📚 Izihlokweni Zokufunda & Amaphepha Okuphasa Adlule",
    "Sesotho": "📚 Ditemane tsa Thuto & Dipampiri tsa Tlhahlobo tsa Mehleng e Fetileng",
}
SUBTITLES = {
    "English": "Select a guide below, choose your language, and press ▶ Play to listen.",
    "isiZulu": "Khetha isiqondiso ngezansi, ukhethe ulimi lwakho, bese ucindezela ▶ Dlala ukuze ulalele.",
    "Sesotho": "Kgetho temane ka tlase, kgetha puo ya hao, ebe o hatsa ▶ Bapala ho utloa.",
}

st.title(TITLES.get(lang["label"], TITLES["English"]))
st.caption(SUBTITLES.get(lang["label"], SUBTITLES["English"]))

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📁 Built-in Guides", "📄 Upload PDF", "❓ How to use"]
)

# ── TAB 1: Built-in study guide samples ───────────────────────────────────────
with tab1:
    GUIDE_HEADINGS = {
        "English": "Choose a built-in study topic",
        "isiZulu": "Khetha isihloko sokufunda esikhona",
        "Sesotho": "Kgetho sehlooho sa thuto se teng",
    }
    st.subheader(GUIDE_HEADINGS.get(lang["label"], GUIDE_HEADINGS["English"]))

    # Sample study guide content — replace with your actual content or DB calls
    SAMPLE_GUIDES = {
        "Mathematics – Algebra Basics": """
Algebra is a branch of mathematics dealing with symbols and the rules for manipulating those symbols.
In algebra, letters represent numbers. For example, if x + 2 = 5, then x = 3.
Key concepts include: variables, constants, expressions, equations, and functions.
To solve a linear equation: isolate the variable on one side of the equals sign.
Example: 3x + 6 = 15 → 3x = 9 → x = 3.
""",
        "Life Sciences – Cell Biology": """
A cell is the basic structural and functional unit of all living organisms.
There are two main types of cells: prokaryotic (no nucleus) and eukaryotic (with nucleus).
Key organelles in eukaryotic cells include: the nucleus (contains DNA), mitochondria (energy production),
the endoplasmic reticulum (protein and lipid synthesis), and the Golgi apparatus (packaging and transport).
Cell division occurs through mitosis (growth/repair) and meiosis (sexual reproduction).
""",
        "Geography – Climate Zones": """
South Africa has a diverse climate with several distinct zones.
The western cape has a Mediterranean climate with dry summers and wet winters.
The interior plateau (highveld) has a semi-arid climate with summer rainfall.
The eastern coast (KwaZulu-Natal) is subtropical and humid.
Climate affects vegetation, agriculture, and human settlement patterns across the country.
""",
    }

    selected_guide = st.selectbox(
        "Study topic · Isihloko · Sehlooho",
        options=list(SAMPLE_GUIDES.keys()),
    )

    if selected_guide:
        english_content = SAMPLE_GUIDES[selected_guide]
        display_content = get_display_text(english_content, lang)

        st.markdown(f"**{selected_guide}**")
        render_text_with_voice(display_content, lang)

# ── TAB 2: PDF upload ──────────────────────────────────────────────────────────
with tab2:
    render_pdf_reader(lang)

# ── TAB 3: Instructions ────────────────────────────────────────────────────────
with tab3:
    HOW_TO = {
        "English": """
### How to use the voice & language feature

1. **Choose your language** in the left sidebar — English, isiZulu, or Sesotho.
2. **Toggle "Read aloud"** in the sidebar to enable text-to-speech.
3. In the *Built-in Guides* tab, pick a study topic and the text will be translated and read to you.
4. In the *Upload PDF* tab, upload any study guide or past exam paper (PDF) — select a section and press play.

> **Tip:** For a slow, clear voice, set Speed to "Slow" in the sidebar.
""",
        "isiZulu": """
### Indlela yokusebenzisa isici sezwi nezilimi

1. **Khetha ulimi lwakho** kusayidbha esekhohlo — isiNgisi, isiZulu, noma Sesotho.
2. **Vula "Funda kakhulu"** kusayidbha ukuze uvule ukufinyelela kwezwi.
3. Kuthebhu ye-*Iziqondiso Ezikhona*, khetha isihloko — umbhalo uzoluhunyushwa uze uwufunde.
4. Kuthebhu yokulayisha i-PDF, layisha noma yiliphi iphepha lezifundo noma elokuxilongwa (PDF).

> **Iseluleko:** Ukuze uzwe izwi elikhuluma kancane, setha isivinini ku-"Slow" kusayidbha.
""",
        "Sesotho": """
### Kamoo ho sebedisa feature ya lentswe le dipuo

1. **Kgetha puo ya hao** karolo ya molao o ka ho le letšehali — Senyesemane, isiZulu, kapa Sesotho.
2. **Hatsetsa "Bala phahameng"** sebading ho kgontsha ho bala ka molomo.
3. Tab ya *Ditemane tse teng*, kgetha sehlooho — mongolo o tla fetolwa ebe o buiswa.
4. Tab ya ho kenya PDF, kenya PDF efe kapa efe ya ditemane tsa thuto kapa dipampiri tsa tlhahlobo.

> **Tlhahiso:** Lentswe le pepeneneng, beha lebelo ho "Slow" sebading.
""",
    }
    st.markdown(HOW_TO.get(lang["label"], HOW_TO["English"]))
