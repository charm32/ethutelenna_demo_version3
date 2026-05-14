# eThute Lenna — Voice & Language Integration Guide
## isiZulu · Sesotho · English read-aloud for study guides and past exam papers

---

## What you get

| Feature | Details |
|---|---|
| 🌐 Language selector | Sidebar dropdown — English / isiZulu / Sesotho |
| 🔊 Read-aloud toggle | Plays any text as MP3 audio in the chosen language |
| 🔄 Translation | Translates English content to isiZulu or Sesotho on the fly |
| 📄 PDF reader | Upload any study guide or past exam paper → select a section → listen |
| ⚡ Free & offline-friendly | Uses gTTS (Google TTS) — no API key required |

---

## Step 1 — Install the new dependencies

```bash
pip install gtts==2.3.2 "googletrans==4.0.0rc1" PyPDF2
```

Or add to your `requirements.txt` (see the included file) and run:

```bash
pip install -r requirements.txt
```

---

## Step 2 — Add the module to your project

Copy `language_voice.py` into your eThute Lenna project root
(the same folder as your `app.py` or main Streamlit file).

```
ethutelenna_app5.0/
├── app.py                  ← your existing main file
├── language_voice.py       ← NEW: drop this in here
├── study_guides_page.py    ← NEW: replace/merge with your guides page
├── requirements.txt        ← updated with new deps
├── .streamlit/
└── study_guides/
```

---

## Step 3 — Plug into your existing pages

For **any** existing Streamlit page, add these three lines:

```python
from language_voice import render_language_sidebar, get_display_text, render_text_with_voice

# At the top of your page:
lang = render_language_sidebar()

# Wherever you display text:
text = get_display_text("Your English content here", lang)
render_text_with_voice(text, lang)
```

That's it. The sidebar will appear on every page automatically.

---

## Step 4 — For your study guides folder

If you already store PDFs in `study_guides/`, you can load them like this:

```python
import os
from language_voice import render_language_sidebar, get_display_text, render_text_with_voice, extract_pdf_text

lang = render_language_sidebar()

guides_dir = "study_guides/"
guide_files = [f for f in os.listdir(guides_dir) if f.endswith(".pdf")]

selected = st.selectbox("Choose a study guide", guide_files)
if selected:
    with open(os.path.join(guides_dir, selected), "rb") as f:
        text = extract_pdf_text(f)
    display = get_display_text(text, lang)
    render_text_with_voice(display, lang)
```

---

## Language codes reference

| Language | gTTS code | Google Translate code |
|---|---|---|
| English | `en` | `en` |
| isiZulu | `zu` | `zu` |
| Sesotho | `st` | `st` |

---

## Upgrading to production-quality voices (optional)

The free `gTTS` library works well for prototyping. For a production app
with higher-quality, more natural-sounding voices in South African languages:

### Option A — Google Cloud TTS (best quality for Zulu)
```bash
pip install google-cloud-texttospeech
```
Requires a GCP project and billing. Supports `zu-ZA` neural voices.

### Option B — Microsoft Azure Speech
```bash
pip install azure-cognitiveservices-speech
```
Supports `zu-ZA` and `st-ZA`. Free tier: 500,000 characters/month.

### Option C — ElevenLabs (most natural-sounding)
Visit https://elevenlabs.io — upload sample audio to clone a voice.
Excellent for custom South African accent voices.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `googletrans` raises `AttributeError` | Use exactly `googletrans==4.0.0rc1` |
| Audio doesn't play in browser | Ensure your Streamlit version is ≥ 1.30 |
| Translation returns English only | Check internet connection (uses Google Translate API) |
| PDF shows no text | The PDF may be scanned (image-based) — needs OCR (install `pytesseract`) |
| gTTS `lang not supported` error | Confirm lang code is exactly `zu` or `st` |

---

## Architecture diagram

```
User selects language (sidebar)
        │
        ▼
Content text (English)
        │
        ├──[if Zulu/Sesotho]──► Google Translate ──► Translated text
        │                                                    │
        └──[if English]──────────────────────────────────────┤
                                                             │
                                                             ▼
                                                    Display on screen
                                                             │
                                              [if voice enabled]
                                                             │
                                                             ▼
                                                gTTS → MP3 audio
                                                             │
                                                             ▼
                                              st.audio() player in browser
```
