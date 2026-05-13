# 🍋 eThute Lenna — v5.0

A Grade 12 study assistant for South African students, built with Streamlit and powered by DeepSeek AI via OpenRouter.

---

## 🚀 What's New in v5.0

| Change | Detail |
|---|---|
| **LLM** | Replaced local Ollama with **DeepSeek V3 via OpenRouter** (cloud, ~$0.14/1M tokens) |
| **Embeddings** | Replaced OllamaEmbeddings with **HuggingFace all-MiniLM-L6-v2** (free, no API key) |
| **Deployment** | Now deployable to **Streamlit Community Cloud** (no local server needed) |

---

## 📁 Project Structure

```
ethutelenna_app5.0/
├── main.py                  # Main Streamlit application
├── config.py                # App configuration & RAG prompt
├── debugger.py              # Logging utilities
├── test_core.py             # Unit tests
├── requirements.txt         # Python dependencies
├── .gitignore
├── .streamlit/
│   └── secrets.toml         # API keys (do NOT commit this file)
└── study_guides/            # Place your PDF study guides here
    └── (add your .pdf files here)
```

---

## ⚙️ Setup

### 1. Get an OpenRouter API Key (free)
Go to [https://openrouter.ai](https://openrouter.ai) → Sign up → API Keys → Create key

### 2. Add your API key

**For local development** — edit `.streamlit/secrets.toml`:
```toml
OPENROUTER_API_KEY = "sk-or-your-key-here"
```

**For Streamlit Cloud** — go to App Settings → Secrets and paste:
```toml
OPENROUTER_API_KEY = "sk-or-your-key-here"
```

### 3. Add study guide PDFs

Place your subject PDF files in the `study_guides/` folder. Name them to match the subject:

```
study_guides/
├── physics.pdf
├── chemistry.pdf
├── mathematics.pdf
├── mathsliteracy.pdf
├── life_sciences.pdf
├── geography.pdf
├── history.pdf
└── english.pdf
```

### 4. Install dependencies (local only)
```bash
pip install -r requirements.txt
```

### 5. Run locally
```bash
streamlit run main.py
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repo, branch `main`, file `main.py`
5. Under **Advanced settings → Secrets**, paste your `OPENROUTER_API_KEY`
6. Click **Deploy**

---

## 🧪 Running Tests
```bash
pytest test_core.py -v
```

---

## 💰 Cost Estimate

| Component | Cost |
|---|---|
| DeepSeek V3 (chat) | ~$0.14 per 1 million tokens |
| HuggingFace embeddings | Free (runs in container) |
| Streamlit Community Cloud | Free |
| **Typical monthly cost** | **< $1 for a student app** |

---

## 📚 Subjects Supported
Physics · Chemistry · Mathematics · Math Literacy · Life Sciences · Geography · History · English
