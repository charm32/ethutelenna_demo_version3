import streamlit as st
import logging
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ── LLM: DeepSeek via OpenRouter (replaces Ollama) ──────────────────────────
# OpenRouter gives access to DeepSeek models through a standard OpenAI-compatible
# API. Sign up free at https://openrouter.ai and get an API key.
# Set the key in Streamlit secrets (secrets.toml) or as an environment variable:
#   OPENROUTER_API_KEY = "sk-or-..."
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import AppConfig
from debugger import DebugLogger

# ── Resolve API key (Streamlit secrets or env var) ───────────────────────────
def get_openrouter_key() -> str:
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return os.environ.get("OPENROUTER_API_KEY", "")

# ── DeepSeek model settings via OpenRouter ───────────────────────────────────
OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"
# deepseek/deepseek-chat is the cheapest full DeepSeek V3 model (~$0.14/1M tokens)
DEEPSEEK_CHAT_MODEL  = "deepseek/deepseek-chat"
# For embeddings we use a free/cheap OpenRouter-compatible embedding model.
# OpenRouter does NOT support embeddings natively, so we use a tiny free
# HuggingFace sentence-transformer via langchain instead (no API key needed).
# ─────────────────────────────────────────────────────────────────────────────

# ── Logo ──
try:
    from logo_data import LOGO_BASE64
    LOGO_IMG_TAG    = f'<img src="data:image/jpeg;base64,{LOGO_BASE64}" style="height:70px;border-radius:10px">'
    LOGO_IMG_TAG_SM = f'<img src="data:image/jpeg;base64,{LOGO_BASE64}" style="height:48px;border-radius:8px">'
    LOGO_IMG_TAG_XS = f'<img src="data:image/jpeg;base64,{LOGO_BASE64}" style="height:36px;border-radius:6px">'
    LOGO_IMG_TAG_LG = f'<img src="data:image/jpeg;base64,{LOGO_BASE64}" style="height:110px;border-radius:12px">'
except ImportError:
    LOGO_BASE64     = ""
    LOGO_IMG_TAG    = "<span style='font-size:2rem'>🍋</span>"
    LOGO_IMG_TAG_SM = "<span style='font-size:1.5rem'>🍋</span>"
    LOGO_IMG_TAG_XS = "<span style='font-size:1.2rem'>🍋</span>"
    LOGO_IMG_TAG_LG = "<span style='font-size:3rem'>🍋</span>"

st.set_page_config(
    page_title="eThute Lenna",
    page_icon="🍋",
    layout="wide",
    initial_sidebar_state="expanded",
)

debug  = DebugLogger(level=logging.INFO)
logger = debug.get_logger(__name__)

USERS_FILE        = "users.json"
TRACKING_FILE     = "tracking.json"
COINS_PER_CORRECT = 10


def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Nunito+Sans:wght@400;600&display=swap');
        html, body, [class*="css"] { font-family: 'Nunito Sans', sans-serif; }
        .stApp { background: linear-gradient(145deg,#f8f4ff 0%,#fdf9ff 50%,#f3eeff 100%); }
        #MainMenu, footer { visibility: hidden; }
        .stDeployButton { display: none; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg,#1a3a5c 0%,#1e5799 60%,#2980b9 100%) !important;
        }
        section[data-testid="stSidebar"] * { color: #e8f4fd !important; }
        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.12) !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
            color: #ffffff !important; border-radius: 10px !important;
            font-family: 'Nunito',sans-serif !important; font-weight: 600 !important;
            width: 100% !important; margin-bottom: 4px !important; transition: all 0.2s !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255,255,255,0.25) !important; transform: translateX(4px) !important;
        }
        .stat-card {
            background: white; border-radius: 14px; padding: 1.2rem 1rem;
            text-align: center; box-shadow: 0 2px 10px rgba(30,87,153,0.08); border-top: 4px solid;
        }
        .stat-card .num   { font-family:'Nunito',sans-serif; font-size:2rem; font-weight:800; color:#1a3a5c; }
        .stat-card .label { font-size:0.78rem; color:#9ca3af; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
        .subject-card {
            background: white; border-radius: 14px; padding: 1.2rem;
            box-shadow: 0 2px 10px rgba(30,87,153,0.07);
            border-left: 5px solid #2980b9; margin-bottom: 0.8rem; transition: all 0.2s;
        }
        .subject-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(30,87,153,0.14); }
        .subject-card h4 { font-family:'Nunito',sans-serif; font-weight:700; color:#1a3a5c; margin:0 0 0.3rem; }
        .subject-card p  { font-size:0.8rem; color:#9ca3af; margin:0; }
        .guide-card {
            background: white; border-radius: 14px; padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 10px rgba(30,87,153,0.07);
            border-left: 5px solid #27ae60; margin-bottom: 0.7rem; transition: all 0.2s;
        }
        .guide-card:hover { transform:translateY(-2px); box-shadow:0 6px 18px rgba(39,174,96,0.15); }
        .guide-card h4 { font-family:'Nunito',sans-serif; font-weight:700; color:#1a5c30; margin:0 0 0.2rem; font-size:1rem; }
        .guide-card p  { font-size:0.8rem; color:#6b7280; margin:0; }
        .track-card {
            background: white; border-radius: 14px; padding: 1.2rem;
            box-shadow: 0 2px 10px rgba(30,87,153,0.07);
            margin-bottom: 0.8rem; border-left: 5px solid #2980b9;
        }
        .track-card h4   { font-family:'Nunito',sans-serif; font-weight:700; color:#1a3a5c; margin:0 0 0.4rem; }
        .track-card .meta{ font-size:0.75rem; color:#9ca3af; margin-bottom:0.5rem; }
        .track-card ul   { margin:0; padding-left:1.2rem; font-size:0.83rem; color:#4b5563; }
        .coin-badge {
            display:inline-block; background:linear-gradient(135deg,#f59e0b,#d97706);
            color:white; border-radius:20px; padding:4px 14px;
            font-family:'Nunito',sans-serif; font-weight:800; font-size:0.9rem;
            box-shadow:0 2px 8px rgba(245,158,11,0.4);
        }
        .coin-total {
            background:linear-gradient(135deg,#fef3c7,#fde68a);
            border:2px solid #f59e0b; border-radius:16px; padding:1.2rem 1.5rem;
            text-align:center; margin-bottom:1.5rem;
        }
        .coin-total .amount   { font-family:'Nunito',sans-serif; font-size:2.8rem; font-weight:800; color:#92400e; }
        .coin-total .subtitle { font-size:0.8rem; color:#b45309; font-weight:600; }
        .stChatMessage { background:transparent !important; border:none !important; }
        [data-testid="stChatMessageContent"] {
            background: white !important; border: 1px solid #dbeafe !important;
            border-radius: 14px !important; padding: 0.85rem 1.1rem !important;
            color: #1e3a5f !important; font-size: 0.92rem !important;
            line-height: 1.75 !important; box-shadow: 0 1px 4px rgba(30,87,153,0.06) !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            background: #eff6ff !important; border-color: #bfdbfe !important;
        }
        .stChatInputContainer {
            background: white !important; border: 1.5px solid #bfdbfe !important;
            border-radius: 14px !important;
        }
        .stChatInputContainer textarea { color: #1e3a5f !important; }
        [data-testid="stChatMessageContent"] h1 { font-size:1.2rem !important; color:#1a3a5c !important; font-family:'Nunito',sans-serif !important; font-weight:800 !important; margin:0.8rem 0 0.4rem !important; }
        [data-testid="stChatMessageContent"] h2 { font-size:1.05rem !important; color:#1e5799 !important; font-family:'Nunito',sans-serif !important; font-weight:700 !important; margin:0.7rem 0 0.3rem !important; }
        [data-testid="stChatMessageContent"] h3 { font-size:0.95rem !important; color:#2980b9 !important; font-family:'Nunito',sans-serif !important; font-weight:700 !important; margin:0.6rem 0 0.3rem !important; }
        [data-testid="stChatMessageContent"] ul  { padding-left:1.3rem !important; margin:0.3rem 0 !important; }
        [data-testid="stChatMessageContent"] li  { margin-bottom:0.3rem !important; font-size:0.9rem !important; }
        [data-testid="stChatMessageContent"] p   { margin:0.3rem 0 !important; font-size:0.9rem !important; }
        [data-testid="stChatMessageContent"] strong { color:#1a3a5c !important; }
        hr { border-color:#dbeafe !important; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-thumb { background:#bfdbfe; border-radius:4px; }
        .stButton > button { font-family:'Nunito',sans-serif !important; font-weight:700 !important; border-radius:10px !important; }
        div[data-testid="stAlert"] { border-radius:10px !important; }
        .stAlert > div { border-radius:10px !important; }
    </style>
    """, unsafe_allow_html=True)


def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def init_session():
    defaults = {
        "page": "login", "user": None,
        "active_subject": None, "active_unit": None,
        "vector_db": None, "db_ready": False,
        "messages": [], "quiz_score": 0,
        "quiz_index": 0, "quiz_active": False, "quiz_answers": [],
        "quiz_saved": False,
        "user_subjects": [], "subject_setup_done": False,
        "study_guides_page": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def register_user(username, password, dob="", school="") -> bool:
    users = load_json(USERS_FILE)
    if username in users:
        return False
    users[username] = {
        "password": hash_password(password),
        "dob": dob, "school": school,
        "created": str(datetime.now()),
        "subjects": [], "coins": 0,
    }
    save_json(USERS_FILE, users)
    return True

def login_user(username, password) -> bool:
    users = load_json(USERS_FILE)
    if username not in users:
        return False
    return users[username]["password"] == hash_password(password)

def get_user_subjects(username) -> list:
    return load_json(USERS_FILE).get(username, {}).get("subjects", [])

def save_user_subjects(username, subjects):
    users = load_json(USERS_FILE)
    if username in users:
        users[username]["subjects"] = subjects
        save_json(USERS_FILE, users)

def get_user_coins(username) -> int:
    return load_json(USERS_FILE).get(username, {}).get("coins", 0)

def add_user_coins(username, amount):
    users = load_json(USERS_FILE)
    if username in users:
        users[username]["coins"] = users[username].get("coins", 0) + amount
        save_json(USERS_FILE, users)

def save_tracking(username, subject, unit, score, tips, total_questions=3):
    data = load_json(TRACKING_FILE)
    if username not in data:
        data[username] = []
    coins_earned = score * COINS_PER_CORRECT
    data[username].append({
        "subject": subject, "unit": unit, "score": score,
        "total_questions": total_questions,
        "tips": tips, "timestamp": datetime.now().strftime("%d %b %Y %H:%M"),
        "coins_earned": coins_earned,
    })
    save_json(TRACKING_FILE, data)
    add_user_coins(username, coins_earned)
    return coins_earned

def get_tracking(username):
    return load_json(TRACKING_FILE).get(username, [])


SUBJECT_CATALOGUE = {
    "Physics": {
        "emoji": "⚡", "color": "#3b82f6",
        "units": [
            {"title": "Mechanics & Motion",     "video": "https://www.youtube.com/watch?v=ZM8ECpBuQYE", "duration": "12 min"},
            {"title": "Waves & Sound",           "video": "https://www.youtube.com/watch?v=dN2-jRBzMCM", "duration": "10 min"},
            {"title": "Electricity & Magnetism", "video": "https://www.youtube.com/watch?v=ruNrdlGpPoQ", "duration": "14 min"},
        ],
        "quiz": [
            {"q": "What is Newton's First Law?",
             "options": ["Objects at rest stay at rest unless acted on","F = ma","Every action has a reaction","Energy is conserved"], "answer": 0,
             "explanation": "Newton's First Law states that an object at rest stays at rest, and an object in motion stays in motion, unless acted on by an external force."},
            {"q": "What is the unit of electric current?",
             "options": ["Volt","Watt","Ampere","Ohm"], "answer": 2,
             "explanation": "Electric current is measured in Amperes (A). Voltage is measured in Volts, power in Watts, and resistance in Ohms."},
            {"q": "What type of wave is sound?",
             "options": ["Transverse","Longitudinal","Electromagnetic","Surface"], "answer": 1,
             "explanation": "Sound is a longitudinal wave — the particles vibrate parallel to the direction the wave travels."},
        ],
        "tips": ["Review Newton's Laws with diagrams","Practice past exam calculations daily","Watch slow-motion videos for wave behaviour"],
    },
    "Chemistry": {
        "emoji": "🧪", "color": "#10b981",
        "units": [
            {"title": "Atomic Structure",     "video": "https://www.youtube.com/watch?v=rz8fHOPGDuI", "duration": "11 min"},
            {"title": "Chemical Bonding",      "video": "https://www.youtube.com/watch?v=QqjcCvzWwww", "duration": "13 min"},
            {"title": "Reactions & Equations", "video": "https://www.youtube.com/watch?v=AXAiLFMjFSc", "duration": "10 min"},
        ],
        "quiz": [
            {"q": "How many electrons does Carbon have?",
             "options": ["4","6","8","12"], "answer": 1,
             "explanation": "Carbon (C) has atomic number 6, which means it has 6 protons and 6 electrons in a neutral atom."},
            {"q": "What bond involves sharing electrons?",
             "options": ["Ionic","Metallic","Covalent","Hydrogen"], "answer": 2,
             "explanation": "Covalent bonds form when atoms share electrons. Ionic bonds transfer electrons between atoms."},
            {"q": "What is the pH of pure water?",
             "options": ["6","7","8","9"], "answer": 1,
             "explanation": "Pure water is neutral with a pH of exactly 7. Below 7 is acidic; above 7 is alkaline/basic."},
        ],
        "tips": ["Draw atomic diagrams by hand","Balance 5 equations per day","Make flashcards for the periodic table"],
    },
    "Mathematics": {
        "emoji": "📐", "color": "#8b5cf6",
        "units": [
            {"title": "Algebra & Functions", "video": "https://www.youtube.com/watch?v=NybHckSEQBI", "duration": "15 min"},
            {"title": "Trigonometry",         "video": "https://www.youtube.com/watch?v=PUB0TaZ7bhA", "duration": "12 min"},
            {"title": "Calculus Basics",      "video": "https://www.youtube.com/watch?v=WUvTyaaNkzM", "duration": "16 min"},
        ],
        "quiz": [
            {"q": "What is the derivative of x²?",
             "options": ["x","2x","x²","2"], "answer": 1,
             "explanation": "Using the power rule: d/dx of xⁿ = n·xⁿ⁻¹. So d/dx of x² = 2x¹ = 2x."},
            {"q": "What does sin(90°) equal?",
             "options": ["0","0.5","1","-1"], "answer": 2,
             "explanation": "sin(90°) = 1. This can be seen on the unit circle where the angle 90° points straight up, giving a y-value of 1."},
            {"q": "What is the quadratic formula used for?",
             "options": ["Finding gradients","Solving quadratic equations","Integration","Geometry"], "answer": 1,
             "explanation": "The quadratic formula x = (-b ± √(b²-4ac)) / 2a is used to find the roots (solutions) of quadratic equations ax²+bx+c=0."},
        ],
        "tips": ["Redo exercises without looking at solutions","Time yourself on past papers","Show all working steps clearly"],
    },
    "Math Literacy": {
        "emoji": "🔢", "color": "#06b6d4",
        "units": [
            {"title": "Finance & Interest", "video": "https://www.youtube.com/watch?v=XNtu55Dh5w0", "duration": "9 min"},
            {"title": "Data & Statistics",  "video": "https://www.youtube.com/watch?v=xxpc-HPKN28", "duration": "11 min"},
            {"title": "Measurement & Maps", "video": "https://www.youtube.com/watch?v=r9NUMbEb0F8", "duration": "8 min"},
        ],
        "quiz": [
            {"q": "What is the simple interest formula?",
             "options": ["P × r × t","P(1+r)^t","P/r×t","P+r+t"], "answer": 0,
             "explanation": "Simple Interest = Principal × rate × time (SI = Prt). This calculates interest without compounding."},
            {"q": "What does the median represent?",
             "options": ["Most common value","Middle value","Average value","Largest value"], "answer": 1,
             "explanation": "The median is the middle value when data is arranged in order. If there are two middle values, take their average."},
            {"q": "What is 25% of 200?",
             "options": ["25","40","50","75"], "answer": 2,
             "explanation": "25% means 25/100 = 0.25. Multiply: 0.25 × 200 = 50."},
        ],
        "tips": ["Apply concepts to real life budgets","Practice reading graphs and charts","Use a calculator to verify mental estimates"],
    },
    "Life Sciences": {
        "emoji": "🌱", "color": "#22c55e",
        "units": [
            {"title": "Cells & Genetics",    "video": "https://www.youtube.com/watch?v=URUJD5NEXC8", "duration": "12 min"},
            {"title": "Evolution & Ecology", "video": "https://www.youtube.com/watch?v=GcjgWov7mTM", "duration": "10 min"},
            {"title": "Human Body Systems",  "video": "https://www.youtube.com/watch?v=Ae4MadKPJC0", "duration": "13 min"},
        ],
        "quiz": [
            {"q": "What is the powerhouse of the cell?",
             "options": ["Nucleus","Ribosome","Mitochondria","Golgi body"], "answer": 2,
             "explanation": "The mitochondria produce ATP (energy) through cellular respiration. That is why it is called the powerhouse of the cell."},
            {"q": "What does DNA stand for?",
             "options": ["Deoxyribonucleic Acid","Double Nucleic Acid","Dynamic Nucleotide Array","Dual Nitrogen Acid"], "answer": 0,
             "explanation": "DNA = Deoxyribonucleic Acid. It is the molecule that carries genetic information in all living organisms."},
            {"q": "What is natural selection?",
             "options": ["Random mutation","Survival of the fittest","Artificial breeding","Genetic engineering"], "answer": 1,
             "explanation": "Natural selection is the process where organisms better adapted to their environment tend to survive and reproduce more. Charles Darwin proposed this theory."},
        ],
        "tips": ["Draw diagrams of cell organelles","Make timelines of evolutionary events","Use mnemonics for body system functions"],
    },
    "Geography": {
        "emoji": "🌍", "color": "#f59e0b",
        "units": [
            {"title": "Climate & Weather",   "video": "https://www.youtube.com/watch?v=x1SgmFa0r04", "duration": "10 min"},
            {"title": "Geomorphology",       "video": "https://www.youtube.com/watch?v=1oCBGCpgqiI", "duration": "11 min"},
            {"title": "Population & Cities", "video": "https://www.youtube.com/watch?v=FACK2knC08E", "duration": "9 min"},
        ],
        "quiz": [
            {"q": "What causes the seasons?",
             "options": ["Distance from the Sun","Earth's tilt on its axis","Moon's gravity","Solar flares"], "answer": 1,
             "explanation": "Seasons are caused by Earth's 23.5° axial tilt. When a hemisphere tilts toward the Sun, it receives more direct sunlight, causing summer."},
            {"q": "What is erosion?",
             "options": ["Building up of land","Wearing away of land","Volcanic activity","Tectonic shift"], "answer": 1,
             "explanation": "Erosion is the wearing away and removal of rock or soil by water, wind, ice, or gravity. It differs from weathering, which breaks rock down in place."},
            {"q": "What is urbanisation?",
             "options": ["Rural farming growth","Movement of people to cities","City population decline","Industrial pollution"], "answer": 1,
             "explanation": "Urbanisation is the process by which an increasing proportion of a population moves from rural areas to urban (city) areas."},
        ],
        "tips": ["Sketch climate graphs from memory","Study SA city case studies","Memorise geomorphological processes"],
    },
    "History": {
        "emoji": "📜", "color": "#a78bfa",
        "units": [
            {"title": "Cold War Era",   "video": "https://www.youtube.com/watch?v=I79TpDe3t2g", "duration": "13 min"},
            {"title": "Apartheid & SA", "video": "https://www.youtube.com/watch?v=SVW3OU-UBvE", "duration": "11 min"},
            {"title": "World War II",   "video": "https://www.youtube.com/watch?v=fo2Rb9h788s", "duration": "14 min"},
        ],
        "quiz": [
            {"q": "When did the Cold War begin?",
             "options": ["1939","1945","1947","1950"], "answer": 2,
             "explanation": "The Cold War is generally considered to have begun in 1947, marked by the Truman Doctrine and growing tension between the USA and USSR after World War II."},
            {"q": "What year did apartheid end in South Africa?",
             "options": ["1990","1994","1996","2000"], "answer": 1,
             "explanation": "Apartheid formally ended in 1994 with South Africa's first democratic elections, won by the ANC under Nelson Mandela."},
            {"q": "Who was SA's first democratically elected president?",
             "options": ["F.W. de Klerk","Desmond Tutu","Nelson Mandela","Walter Sisulu"], "answer": 2,
             "explanation": "Nelson Mandela became South Africa's first democratically elected president in 1994, after spending 27 years in prison."},
        ],
        "tips": ["Create timelines of major events","Practise essay structure","Link causes and effects for each event"],
    },
    "English": {
        "emoji": "📖", "color": "#ec4899",
        "units": [
            {"title": "Literature & Poetry", "video": "https://www.youtube.com/watch?v=JwhouCNq-Fc", "duration": "10 min"},
            {"title": "Writing Skills",      "video": "https://www.youtube.com/watch?v=JrU1ADCiRH4", "duration": "9 min"},
            {"title": "Language & Grammar",  "video": "https://www.youtube.com/watch?v=vFQlWCJ1Mm4", "duration": "8 min"},
        ],
        "quiz": [
            {"q": "What is a metaphor?",
             "options": ["A comparison using 'like' or 'as'","An indirect comparison without 'like' or 'as'","A repeated sound","An exaggeration"], "answer": 1,
             "explanation": "A metaphor directly compares two things by saying one IS the other (e.g. 'Life is a journey'). A simile uses 'like' or 'as'."},
            {"q": "What is the purpose of a topic sentence?",
             "options": ["To conclude a paragraph","To introduce the main idea of a paragraph","To give examples","To add detail"], "answer": 1,
             "explanation": "A topic sentence introduces the main idea of a paragraph. It tells the reader what the paragraph will be about."},
            {"q": "What tense describes past actions still relevant now?",
             "options": ["Simple past","Past perfect","Present perfect","Future tense"], "answer": 2,
             "explanation": "Present perfect (e.g. 'I have studied') describes actions that happened in the past but are still relevant or connected to the present."},
        ],
        "tips": ["Read a passage aloud every day","Practise introductions and conclusions","Keep a vocabulary journal"],
    },
}

ALL_SUBJECT_NAMES = list(SUBJECT_CATALOGUE.keys())


# ── Flexible PDF-to-subject matching ──
SUBJECT_PDF_MAP = {
    "Physics":       ["physics"],
    "Chemistry":     ["chemistry"],
    "Mathematics":   ["mathematics", "maths grade", "math grade"],
    "Math Literacy": ["maths_lit", "maths lit", "math lit", "mathematical literacy", "mathslit"],
    "Life Sciences": ["life science", "biology", "life_science"],
    "Geography":     ["geography", "geo"],
    "History":       ["history"],
    "English":       ["english"],
}

def get_available_study_guides() -> dict:
    guides_dir = Path("study_guides")
    guides_dir.mkdir(exist_ok=True)
    return {
        pdf.stem.replace("_", " ").replace("-", " "): str(pdf)
        for pdf in sorted(guides_dir.glob("*.pdf"))
    }

def find_guide_for_subject(subject: str) -> str | None:
    keywords = SUBJECT_PDF_MAP.get(subject, [subject.lower()])
    guides   = get_available_study_guides()
    for name, path in guides.items():
        name_lower = name.lower().replace("_", " ").replace("-", " ")
        for kw in keywords:
            if kw in name_lower or name_lower in kw:
                return path
    for name, path in guides.items():
        if subject.lower() in name.lower():
            return path
    return None


# ── Embeddings: free HuggingFace sentence-transformer (no API key needed) ────
# This runs locally inside the Streamlit Cloud container — no cost at all.
# The model is tiny (~90 MB) and fast enough for a study app.
@st.cache_resource(show_spinner=False)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_subject_db(subject_name: str, pdf_path: str):
    """Load or build a Chroma vector DB from a study guide PDF."""
    collection_name = f"ethute_{subject_name.replace(' ','_').lower()}"
    chroma_path     = f"chroma_db/{collection_name}"
    try:
        embeddings = get_embeddings()
        if os.path.exists(chroma_path) and os.listdir(chroma_path):
            logger.info(f"Reusing existing DB: {collection_name}")
            return Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=chroma_path,
            )
        logger.info(f"Building DB for {subject_name} from {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs   = loader.load()
        if not docs:
            logger.error(f"No pages loaded from {pdf_path}")
            return None
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        for doc in chunks:
            doc.page_content = doc.page_content[:AppConfig.MAX_CHARS]
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=chroma_path,
        )
        logger.info(f"DB built: {len(chunks)} chunks for {subject_name}")
        return db
    except Exception as e:
        logger.error(f"DB error for {subject_name}: {e}")
        return None


RAG_PROMPT = AppConfig.RAG_PROMPT


def get_deepseek_llm(temperature: float = 0):
    """
    Return a LangChain ChatOpenAI client pointed at OpenRouter's DeepSeek V3.
    DeepSeek V3 (deepseek/deepseek-chat) on OpenRouter costs ~$0.14 per 1M tokens
    — roughly free for a student app with light usage.
    """
    api_key = get_openrouter_key()
    if not api_key:
        st.error(
            "⚠️ No OpenRouter API key found. "
            "Add OPENROUTER_API_KEY to your Streamlit secrets or environment variables. "
            "Sign up free at https://openrouter.ai"
        )
        st.stop()
    return ChatOpenAI(
        model=DEEPSEEK_CHAT_MODEL,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        default_headers={
            # OpenRouter requires these headers to identify your app
            "HTTP-Referer": "https://ethutelenna.com",
            "X-Title": "eThute Lenna",
        },
    )


def clean_answer(text: str) -> str:
    import re
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if re.search(r'\[.*?(write|short|one |two |three |explain|heading|tip|example|summar|relevant).*?\]', line, re.IGNORECASE):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    if len(result) < 30:
        return "📚 I could not find a clear answer in your study guide for that question. Please try rephrasing or ask something else."
    return result


def answer_question_from_guide(question: str, vector_db) -> str:
    """Run RAG chain: retrieve from PDF chunks then generate answer with DeepSeek."""
    try:
        llm       = get_deepseek_llm(temperature=0)
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": AppConfig.RETRIEVAL_K},
        )
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        chain  = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        raw = chain.invoke(question)
        return clean_answer(raw)
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return f"⚠️ Something went wrong: {e}"


QUIZ_EXPLAIN_PROMPT = """You are eThute Lenna, a Grade 12 study assistant.
Using ONLY the context from the study guide below, explain in 3-5 short bullet points why the correct answer is correct.
Keep each bullet to one sentence. Start with "## Why {correct_answer} is correct".
If the study guide does not cover this, use this pre-written explanation: {fallback}

Study Guide Context: {context}
Question: {question}
Correct Answer: {correct_answer}
Explanation (bullet points only):"""


def explain_quiz_answer(question: str, correct_answer: str, fallback: str, vector_db) -> str:
    if vector_db is None:
        return fallback
    try:
        llm       = get_deepseek_llm(temperature=0)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        prompt    = ChatPromptTemplate.from_template(QUIZ_EXPLAIN_PROMPT)
        chain     = (
            {"context": retriever,
             "question": lambda _: question,
             "correct_answer": lambda _: correct_answer,
             "fallback": lambda _: fallback}
            | prompt | llm | StrOutputParser()
        )
        return chain.invoke(question)
    except Exception:
        return fallback


# ==================== PAGES ====================

def page_login():
    inject_css()
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown(f"<div style='text-align:center;margin-bottom:0.5rem'>{LOGO_IMG_TAG_LG}</div>",
                    unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;font-family:Nunito;color:#1a3a5c;margin:0'>eThute Lenna</h2>",
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#9ca3af;font-size:0.85rem;margin-bottom:1.5rem'>Grade 12 Lesson Guide</p>",
                    unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", use_container_width=True, type="primary"):
                if login_user(username, password):
                    st.session_state.user = username
                    saved = get_user_subjects(username)
                    st.session_state.user_subjects      = saved
                    st.session_state.subject_setup_done = len(saved) > 0
                    st.session_state.page = "dashboard" if st.session_state.subject_setup_done else "subject_setup"
                    st.rerun()
                else:
                    st.markdown("""
                        <div style='background:#fef3c7;border:1px solid #f59e0b;border-radius:10px;
                            padding:10px 14px;font-size:0.85rem;color:#92400e;margin-top:8px'>
                            ⚠️ Incorrect username or password. Please try again.
                        </div>
                    """, unsafe_allow_html=True)
        with tab2:
            new_user = st.text_input("Username", key="reg_user")
            dob      = st.date_input("Date of Birth", key="reg_dob",
                                     min_value=datetime(1990,1,1).date(),
                                     max_value=datetime(2015,12,31).date())
            school   = st.text_input("School Name", key="reg_school")
            new_pass = st.text_input("Password", type="password", key="reg_pass")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if new_user and new_pass and school:
                    if register_user(new_user, new_pass, str(dob), school):
                        st.success("✅ Account created! Please log in.")
                    else:
                        st.markdown("""
                            <div style='background:#fef3c7;border:1px solid #f59e0b;border-radius:10px;
                                padding:10px 14px;font-size:0.85rem;color:#92400e;margin-top:8px'>
                                ⚠️ Username already taken. Please choose another.
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Please fill in all fields.")


def page_subject_setup():
    inject_css()
    _, col, _ = st.columns([0.4, 2, 0.4])
    with col:
        st.markdown(f"""
            <div style='text-align:center;margin-bottom:1.2rem'>
                {LOGO_IMG_TAG_SM}
                <h2 style='font-family:Nunito;font-weight:800;color:#1a3a5c;margin:0.5rem 0 0.2rem'>
                    Choose Your Subjects
                </h2>
                <p style='color:#1e5799;font-size:0.88rem;margin-bottom:0.5rem'>
                    Select the subjects you are studying this year.<br>
                    You can change your selection anytime from the sidebar.
                </p>
            </div>
        """, unsafe_allow_html=True)
        if "pending_subjects" not in st.session_state:
            st.session_state.pending_subjects = list(st.session_state.user_subjects)
        st.markdown("<p style='font-family:Nunito;color:#1a3a5c;font-weight:700;font-size:0.92rem'>📚 Tap a subject to select / deselect it:</p>",
                    unsafe_allow_html=True)
        rows = [ALL_SUBJECT_NAMES[i:i+3] for i in range(0, len(ALL_SUBJECT_NAMES), 3)]
        for row in rows:
            rcols = st.columns(3)
            for j, subj in enumerate(row):
                info        = SUBJECT_CATALOGUE[subj]
                is_selected = subj in st.session_state.pending_subjects
                label       = f"{'✅ ' if is_selected else ''}{info['emoji']} {subj}"
                with rcols[j]:
                    if st.button(label, key=f"sel_{subj}", use_container_width=True,
                                 type="primary" if is_selected else "secondary"):
                        if is_selected:
                            st.session_state.pending_subjects.remove(subj)
                        else:
                            st.session_state.pending_subjects.append(subj)
                        st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        n = len(st.session_state.pending_subjects)
        if n > 0:
            names = ", ".join(st.session_state.pending_subjects)
            st.markdown(f"<p style='color:#1e5799;font-weight:700;font-size:0.88rem'>✅ {n} selected: {names}</p>",
                        unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background:#fef3c7;border:1px solid #f59e0b;border-radius:10px;
                    padding:8px 14px;font-size:0.85rem;color:#92400e'>
                    ⚠️ Please select at least one subject.
                </div>
            """, unsafe_allow_html=True)
        if st.button("✅ Save My Subjects & Continue", use_container_width=True,
                     type="primary", disabled=(n == 0)):
            st.session_state.user_subjects      = list(st.session_state.pending_subjects)
            st.session_state.subject_setup_done = True
            save_user_subjects(st.session_state.user, st.session_state.user_subjects)
            del st.session_state["pending_subjects"]
            st.session_state.page = "dashboard"
            st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown(f"<div style='text-align:center;padding:0.5rem 0 0.2rem'>{LOGO_IMG_TAG_SM}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;font-size:0.75rem;opacity:0.75;margin:0 0 0.1rem'>👤 {st.session_state.user}</p>",
                    unsafe_allow_html=True)
        coins = get_user_coins(st.session_state.user)
        st.markdown(f"<p style='text-align:center;font-size:0.82rem;margin:0 0 0.3rem'>🪙 <strong>{coins} coins</strong></p>",
                    unsafe_allow_html=True)
        st.divider()
        st.markdown("<p style='font-size:0.7rem;letter-spacing:1px;opacity:0.65;margin-bottom:4px'>NAVIGATION</p>",
                    unsafe_allow_html=True)
        if st.button("🏠  Dashboard"):
            st.session_state.page = "dashboard"; st.rerun()
        if st.button("📊  My Progress"):
            st.session_state.page = "tracking"; st.rerun()
        if st.button("📂  Study Guides"):
            st.session_state.page = "study_guides"; st.rerun()
        if st.button("📝  Previous Papers"):
            st.session_state.page = "previous_papers"; st.rerun()
        if st.button("⚙️  My Subjects"):
            if "pending_subjects" in st.session_state:
                del st.session_state["pending_subjects"]
            st.session_state.page = "subject_setup"; st.rerun()
        st.divider()
        st.markdown("<p style='font-size:0.7rem;letter-spacing:1px;opacity:0.65;margin-bottom:4px'>MY SUBJECTS</p>",
                    unsafe_allow_html=True)
        user_subjects = st.session_state.get("user_subjects", [])
        if not user_subjects:
            st.markdown("<p style='font-size:0.75rem;opacity:0.5;padding:4px 8px'>No subjects yet — click ⚙️ My Subjects</p>",
                        unsafe_allow_html=True)
        for subject in user_subjects:
            info  = SUBJECT_CATALOGUE.get(subject, {})
            emoji = info.get("emoji", "📚")
            if st.button(f"{emoji}  {subject}", key=f"nav_{subject}"):
                _open_subject(subject)
        st.divider()
        if st.button("🚪  Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def _open_subject(subject: str):
    st.session_state.active_subject = subject
    st.session_state.page           = "subject"
    st.session_state.messages       = []
    st.session_state.db_ready       = False
    st.session_state.vector_db      = None
    pdf_path = find_guide_for_subject(subject)
    if pdf_path:
        with st.spinner(f"Loading {subject} study guide..."):
            db = load_subject_db(subject, pdf_path)
            st.session_state.vector_db = db
            st.session_state.db_ready  = db is not None
    st.rerun()


def page_dashboard():
    inject_css()
    render_sidebar()
    coins    = get_user_coins(st.session_state.user)
    tracking = get_tracking(st.session_state.user)
    quizzes_done     = len(tracking)
    subjects_visited = len(set(t["subject"] for t in tracking)) if tracking else 0
    avg_score        = round(sum(t["score"] / max(t.get("total_questions", 3), 1) for t in tracking) / quizzes_done * 100) if quizzes_done else 0

    st.markdown(f"""
        <div style='display:flex;align-items:center;gap:14px;background:white;
            border-radius:16px;padding:1rem 1.5rem;
            box-shadow:0 2px 12px rgba(30,87,153,0.08);margin-bottom:1.5rem'>
            {LOGO_IMG_TAG}
            <div>
                <h1 style='font-family:Nunito;font-size:1.4rem;font-weight:800;color:#1a3a5c;margin:0'>eThute Lenna</h1>
                <p style='font-size:0.78rem;color:#9ca3af;margin:0'>Personal Study Assistant</p>
            </div>
            <div style='margin-left:auto'>
                <div class='coin-badge'>🪙 {coins} coins</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label, color in [
        (c1, subjects_visited, "Subjects Explored", "#3b82f6"),
        (c2, quizzes_done,     "Quizzes Completed",  "#10b981"),
        (c3, f"{avg_score}%",  "Average Score",      "#8b5cf6"),
        (c4, f"🪙{coins}",     "Gold Coins",          "#f59e0b"),
    ]:
        with col:
            st.markdown(f'<div class="stat-card" style="border-color:{color}"><div class="num">{num}</div><div class="label">{label}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    user_subjects = st.session_state.get("user_subjects", [])
    if not user_subjects:
        st.markdown("""
            <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
                padding:1rem 1.2rem;color:#1e3a5f;font-size:0.9rem'>
                👆 You have not selected any subjects yet.
                Click <strong>⚙️ My Subjects</strong> in the sidebar to get started.
            </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("<h3 style='font-family:Nunito;color:#1a3a5c;font-size:1rem;font-weight:700'>My Subjects</h3>",
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, subject in enumerate(user_subjects):
        info  = SUBJECT_CATALOGUE.get(subject, {})
        color = info.get("color", "#3b82f6")
        emoji = info.get("emoji", "📚")
        with cols[i % 2]:
            st.markdown(f"""
                <div class="subject-card" style="border-left-color:{color}">
                    <h4>{emoji} {subject}</h4>
                    <p>{len(info.get('units',[]))} study units · {len(info.get('quiz',[]))} quiz questions</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {emoji} {subject}", key=f"open_{subject}"):
                _open_subject(subject)


def page_study_guides():
    inject_css()
    render_sidebar()
    st.markdown("""
        <h2 style='font-family:Nunito;font-weight:800;color:#1a3a5c;margin-bottom:0.2rem'>
            📂 Study Guides
        </h2>
        <p style='color:#6b7280;font-size:0.85rem;margin-bottom:1.2rem'>
            All study guides available. Answers in Ask a Question come directly from these PDFs.
        </p>
    """, unsafe_allow_html=True)
    guides = get_available_study_guides()
    if not guides:
        st.info("No study guides found. Add PDF files to the study_guides/ folder.")
        return
    st.markdown(f"<p style='color:#1e5799;font-weight:700;font-size:0.88rem'>✅ {len(guides)} study guide(s) loaded and ready:</p>",
                unsafe_allow_html=True)
    border_colors = ["#3b82f6","#10b981","#8b5cf6","#f59e0b","#06b6d4","#22c55e","#a78bfa","#ec4899"]
    for idx, (name, path) in enumerate(guides.items()):
        color = border_colors[idx % len(border_colors)]
        try:
            filesize = f"{round(os.path.getsize(path) / 1_048_576, 1)} MB"
        except Exception:
            filesize = ""
        st.markdown(f"""
            <div class="guide-card" style="border-left-color:{color}">
                <h4>📄 {name}</h4>
                <p>📁 {path} &nbsp;·&nbsp; 💾 {filesize}</p>
            </div>
        """, unsafe_allow_html=True)


def page_subject():
    inject_css()
    render_sidebar()
    subject = st.session_state.active_subject
    if not subject or subject not in SUBJECT_CATALOGUE:
        st.session_state.page = "dashboard"; st.rerun()

    data  = SUBJECT_CATALOGUE[subject]
    emoji = data.get("emoji", "📚")
    color = data.get("color", "#3b82f6")

    st.markdown(f"""
        <h2 style='font-family:Nunito;font-weight:800;color:#1a3a5c;margin-bottom:0.2rem'>{emoji} {subject}</h2>
        <p style='color:#9ca3af;font-size:0.85rem;margin-bottom:1rem'>Grade 12 · {len(data["units"])} Study Units</p>
    """, unsafe_allow_html=True)

    tab_units, tab_chat, tab_quiz = st.tabs(["🎬 Animated Study Units", "❓ Ask a Question", "📝 Quiz"])

    with tab_units:
        st.markdown("<p style='color:#6b7280;font-size:0.85rem;margin-bottom:1rem'>Watch each lesson then take the quiz.</p>",
                    unsafe_allow_html=True)
        for i, unit in enumerate(data["units"]):
            with st.expander(f"▶️  Unit {i+1}: {unit['title']}  ·  {unit['duration']}"):
                st.video(unit["video"])
                st.markdown(f"<p style='font-size:0.8rem;color:#9ca3af'>Duration: {unit['duration']}</p>",
                            unsafe_allow_html=True)
                if st.button(f"📝 Take Quiz for Unit {i+1}", key=f"quiz_btn_{i}"):
                    st.session_state.active_unit  = unit["title"]
                    st.session_state.quiz_active  = True
                    st.session_state.quiz_index   = 0
                    st.session_state.quiz_score   = 0
                    st.session_state.quiz_answers = []
                    st.session_state.quiz_saved   = False
                    st.session_state.page         = "quiz"
                    st.rerun()

    with tab_chat:
        if not st.session_state.db_ready:
            pdf_path = find_guide_for_subject(subject)
            if pdf_path:
                with st.spinner(f"📖 Loading your {subject} study guide..."):
                    db = load_subject_db(subject, pdf_path)
                    st.session_state.vector_db = db
                    st.session_state.db_ready  = db is not None

        for msg in st.session_state.messages:
            avatar = "🍋" if msg["role"] == "assistant" else "🎓"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        if user_input := st.chat_input(f"Ask anything about {subject}..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="🎓"):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar="🍋"):
                if st.session_state.db_ready:
                    with st.spinner("🔍 Searching your study guide..."):
                        response = answer_question_from_guide(
                            user_input, st.session_state.vector_db
                        )
                else:
                    response = (
                        "📚 Your study guide is still loading. "
                        "Please wait a moment and ask again."
                    )
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    with tab_quiz:
        max_coins = len(data["quiz"]) * COINS_PER_CORRECT
        st.markdown(f"""
            <div style='background:#f0fdf4;border-left:4px solid #22c55e;
                border-radius:10px;padding:10px 14px;margin-bottom:1rem'>
                <p style='margin:0;font-size:0.85rem;color:#15803d'>
                    📝 Answer all questions and earn 🪙 gold coins!<br>
                    <strong>Earn up to 🪙 {max_coins} coins for this quiz.</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("▶ Start Full Quiz", type="primary"):
            st.session_state.active_unit  = "Full Quiz"
            st.session_state.quiz_active  = True
            st.session_state.quiz_index   = 0
            st.session_state.quiz_score   = 0
            st.session_state.quiz_answers = []
            st.session_state.quiz_saved   = False
            st.session_state.page         = "quiz"
            st.rerun()


def page_quiz():
    inject_css()
    render_sidebar()
    subject   = st.session_state.active_subject
    unit      = st.session_state.active_unit
    questions = SUBJECT_CATALOGUE[subject]["quiz"]
    idx       = st.session_state.quiz_index
    emoji     = SUBJECT_CATALOGUE[subject]["emoji"]
    color     = SUBJECT_CATALOGUE[subject]["color"]

    st.markdown(f"""
        <h2 style='font-family:Nunito;font-weight:800;color:#1a3a5c'>{emoji} {subject} — Quiz</h2>
        <p style='color:#9ca3af;font-size:0.85rem'>{unit} · 🪙 {COINS_PER_CORRECT} coins per correct answer</p>
    """, unsafe_allow_html=True)

    if idx < len(questions):
        st.progress(idx / len(questions), text=f"Question {idx+1} of {len(questions)}")
        q = questions[idx]
        st.markdown(f"""
            <div style='background:white;border-radius:14px;padding:1.5rem;
                border-left:5px solid {color};
                box-shadow:0 2px 10px rgba(30,87,153,0.08);margin:1rem 0'>
                <h3 style='font-family:Nunito;color:#1a3a5c;font-size:1.05rem'>{q["q"]}</h3>
            </div>
        """, unsafe_allow_html=True)
        answer = st.radio("Choose your answer:", q["options"], key=f"quiz_q_{idx}", index=None)
        if st.button("Submit Answer ➜", type="primary"):
            if answer is None:
                st.warning("Please select an answer first.")
            else:
                correct      = q["options"].index(answer) == q["answer"]
                correct_text = q["options"][q["answer"]]
                fallback_exp = q.get("explanation", f"{correct_text} is the correct answer.")
                if correct:
                    st.session_state.quiz_score += 1
                    st.markdown(f"""
                        <div style='background:#f0fdf4;border-left:4px solid #22c55e;
                            border-radius:10px;padding:10px 14px;font-size:0.88rem;color:#15803d'>
                            ✅ <strong>Correct!</strong> +🪙 {COINS_PER_CORRECT} coins
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='background:#fef3c7;border-left:4px solid #f59e0b;
                            border-radius:10px;padding:10px 14px;font-size:0.88rem;color:#92400e'>
                            ⚠️ <strong>Incorrect.</strong> The correct answer is: <strong>{correct_text}</strong>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div style='background:#eff6ff;border-left:4px solid {color};
                        border-radius:10px;padding:10px 14px;font-size:0.85rem;
                        color:#1e3a5f;margin-top:8px'>
                        {fallback_exp}
                    </div>
                """, unsafe_allow_html=True)
                st.session_state.quiz_answers.append({"q": q["q"], "correct": correct})
                st.session_state.quiz_index += 1
                st.rerun()
    else:
        score = st.session_state.quiz_score
        total = len(questions)
        pct   = round(score / total * 100)
        tips  = SUBJECT_CATALOGUE[subject].get("tips", [])
        if not st.session_state.get("quiz_saved", False):
            coins_earned = save_tracking(st.session_state.user, subject, unit, score, tips, total_questions=total)
            st.session_state.quiz_saved   = True
            st.session_state.coins_earned = coins_earned
        else:
            coins_earned = st.session_state.get("coins_earned", 0)
        total_coins = get_user_coins(st.session_state.user)
        medal = "🥇" if pct == 100 else ("🥈" if pct >= 67 else "🥉")
        msg   = ("Perfect score! Outstanding!" if pct == 100
                 else ("Great effort! Keep it up!" if pct >= 67
                       else "Keep practising — you have got this!"))
        st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1e5799,#1a3a5c);border-radius:20px;
                padding:2rem;text-align:center;color:white;margin:1rem 0'>
                <div style='font-size:3.5rem'>{medal}</div>
                <div style='font-size:2.8rem;font-weight:800;font-family:Nunito'>{pct}%</div>
                <div style='font-size:1rem;opacity:0.85'>{score} out of {total} correct</div>
                <div style='font-size:0.85rem;opacity:0.7;margin-top:0.3rem'>{msg}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class='coin-total'>
                <div style='font-size:2rem'>🪙</div>
                <div class='amount'>+{coins_earned} coins earned!</div>
                <div class='subtitle'>Total wallet: {total_coins} coins</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<h4 style='font-family:Nunito;color:#1a3a5c'>💡 How to improve:</h4>", unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f"""
                <div style='background:#eff6ff;border-radius:8px;padding:6px 12px;
                    margin-bottom:6px;font-size:0.85rem;color:#1e3a5f'>• {tip}</div>
            """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔁 Retake Quiz"):
                st.session_state.quiz_index   = 0
                st.session_state.quiz_score   = 0
                st.session_state.quiz_answers = []
                st.session_state.quiz_saved   = False
                st.rerun()
        with c2:
            if st.button("📊 See My Progress"):
                st.session_state.page = "tracking"; st.rerun()


def page_tracking():
    inject_css()
    render_sidebar()
    import pandas as pd
    user    = st.session_state.user
    records = get_tracking(user)
    coins   = get_user_coins(user)

    st.markdown(f"""
        <h2 style='font-family:Nunito;font-weight:800;color:#1a3a5c'>📊 My Progress Dashboard</h2>
        <p style='color:#9ca3af;font-size:0.85rem'>Learning journey for: <strong>{user}</strong></p>
    """, unsafe_allow_html=True)
    st.markdown(f"""
        <div class='coin-total'>
            <div style='font-size:2.2rem'>🪙</div>
            <div class='amount'>{coins}</div>
            <div class='subtitle'>Gold Coins — earned by completing quizzes correctly</div>
        </div>
    """, unsafe_allow_html=True)

    if not records:
        st.markdown("""
            <div style='text-align:center;padding:3rem;color:#9ca3af'>
                <div style='font-size:2.5rem'>📋</div>
                No quiz results yet. Complete a quiz to see your progress here.
            </div>
        """, unsafe_allow_html=True)
        return

    total         = len(records)
    subjects_done = list(set(r["subject"] for r in records))
    avg           = round(sum(r["score"] / max(r.get("total_questions", 3), 1) for r in records) / total * 100)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label, color in [
        (c1, len(subjects_done), "Subjects",    "#3b82f6"),
        (c2, total,              "Quizzes Done", "#10b981"),
        (c3, f"{avg}%",          "Avg Score",    "#8b5cf6"),
        (c4, f"🪙{coins}",       "Coins Total",  "#f59e0b"),
    ]:
        with col:
            st.markdown(f'<div class="stat-card" style="border-color:{color}"><div class="num">{num}</div><div class="label">{label}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    chart_data = []
    for r in records:
        pct = round(r["score"] / max(r.get("total_questions", 3), 1) * 100)
        chart_data.append({
            "Quiz":         f"{r['subject'][:9]} ({r['timestamp'][:6]})",
            "Score (%)":    pct,
            "Subject":      r["subject"],
            "Coins Earned": r.get("coins_earned", 0),
        })
    df = pd.DataFrame(chart_data)
    st.markdown("<h3 style='font-family:Nunito;color:#1a3a5c;font-size:1rem;font-weight:700'>📈 Quiz Score History</h3>", unsafe_allow_html=True)
    st.bar_chart(df.set_index("Quiz")["Score (%)"], color="#3b82f6", height=280)
    st.markdown("<h3 style='font-family:Nunito;color:#1a3a5c;font-size:1rem;font-weight:700;margin-top:1rem'>🪙 Gold Coins Per Quiz</h3>", unsafe_allow_html=True)
    st.bar_chart(df.set_index("Quiz")["Coins Earned"], color="#f59e0b", height=200)
    st.markdown("<h3 style='font-family:Nunito;color:#1a3a5c;font-size:1rem;font-weight:700;margin-top:1rem'>📚 Average Score by Subject</h3>", unsafe_allow_html=True)
    subject_avgs = df.groupby("Subject")["Score (%)"].mean().round(1)
    st.bar_chart(subject_avgs, color="#10b981", height=220)

    st.markdown("<br><h4 style='font-family:Nunito;color:#1a3a5c'>🗂️ Quiz History</h4>", unsafe_allow_html=True)
    for r in reversed(records):
        pct       = round(r["score"] / max(r.get("total_questions", 3), 1) * 100)
        tips_html = "".join(f"<li>{t}</li>" for t in r["tips"])
        earned    = r.get("coins_earned", 0)
        color     = SUBJECT_CATALOGUE.get(r["subject"], {}).get("color", "#3b82f6")
        st.markdown(f"""
            <div class="track-card" style="border-left-color:{color}">
                <h4>{r["subject"]} — {r["unit"]}</h4>
                <div class="meta">🗓 {r["timestamp"]} &nbsp;·&nbsp; Score: <strong>{r["score"]}/3 ({pct}%)</strong>
                &nbsp;·&nbsp; 🪙 <strong>+{earned} coins</strong></div>
                <strong style='font-size:0.82rem;color:#1e5799'>💡 How to improve:</strong>
                <ul>{tips_html}</ul>
            </div>
        """, unsafe_allow_html=True)


def page_previous_papers():
    inject_css()
    render_sidebar()

    st.markdown("""
        <h2 style='font-family:Nunito;font-weight:800;color:#1a3a5c;margin-bottom:0.2rem'>
            📝 Previous Question Papers
        </h2>
        <p style='color:#6b7280;font-size:0.85rem;margin-bottom:1.5rem'>
            Download and practise with official past exam papers to prepare for your Grade 12 finals.
        </p>
    """, unsafe_allow_html=True)

    PAPERS = {
        "Physics": {
            "emoji": "⚡", "color": "#3b82f6",
            "papers": [
                {"year": "2023", "label": "Physical Sciences P1 (Physics) 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2022", "label": "Physical Sciences P1 (Physics) 2022", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2021", "label": "Physical Sciences P1 (Physics) 2021", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
            ],
        },
        "Chemistry": {
            "emoji": "🧪", "color": "#10b981",
            "papers": [
                {"year": "2023", "label": "Physical Sciences P2 (Chemistry) 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2022", "label": "Physical Sciences P2 (Chemistry) 2022", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2021", "label": "Physical Sciences P2 (Chemistry) 2021", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
            ],
        },
        "Mathematics": {
            "emoji": "📐", "color": "#8b5cf6",
            "papers": [
                {"year": "2023", "label": "Mathematics P1 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2023", "label": "Mathematics P2 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2022", "label": "Mathematics P1 2022", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2022", "label": "Mathematics P2 2022", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
            ],
        },
        "Math Literacy": {
            "emoji": "🔢", "color": "#06b6d4",
            "papers": [
                {"year": "2023", "label": "Mathematical Literacy P1 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2023", "label": "Mathematical Literacy P2 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2022", "label": "Mathematical Literacy P1 2022", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
            ],
        },
        "Life Sciences": {
            "emoji": "🌱", "color": "#22c55e",
            "papers": [
                {"year": "2023", "label": "Life Sciences P1 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2023", "label": "Life Sciences P2 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2022", "label": "Life Sciences P1 2022", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
            ],
        },
        "Geography": {
            "emoji": "🌍", "color": "#f59e0b",
            "papers": [
                {"year": "2023", "label": "Geography P1 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2023", "label": "Geography P2 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2022", "label": "Geography P1 2022", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
            ],
        },
        "History": {
            "emoji": "📜", "color": "#a78bfa",
            "papers": [
                {"year": "2023", "label": "History P1 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2023", "label": "History P2 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2022", "label": "History P1 2022", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
            ],
        },
        "English": {
            "emoji": "📖", "color": "#ec4899",
            "papers": [
                {"year": "2023", "label": "English Home Language P1 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2023", "label": "English Home Language P2 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
                {"year": "2023", "label": "English Home Language P3 2023", "url": "https://www.education.gov.za/Curriculum/NationalSeniorCertificate/NSCPastExaminationpapers.aspx"},
            ],
        },
    }

    user_subjects = st.session_state.get("user_subjects", [])
    subjects_to_show = user_subjects if user_subjects else list(PAPERS.keys())

    st.markdown("""
        <div style='background:linear-gradient(135deg,#eff6ff,#dbeafe);
            border:1px solid #93c5fd;border-radius:12px;
            padding:0.9rem 1.2rem;font-size:0.83rem;color:#1e3a5f;margin-bottom:1.5rem'>
            🏛️ <strong>All papers are sourced from the official Department of Basic Education website.</strong><br>
            Clicking a paper will open the DBE past papers page where you can download the PDF.
        </div>
    """, unsafe_allow_html=True)

    for subject in subjects_to_show:
        info = PAPERS.get(subject)
        if not info:
            continue
        emoji  = info["emoji"]
        color  = info["color"]
        papers = info["papers"]

        st.markdown(f"""
            <div style='background:white;border-radius:14px;padding:1.2rem 1.4rem;
                border-left:5px solid {color};
                box-shadow:0 2px 10px rgba(30,87,153,0.07);margin-bottom:1rem'>
                <h4 style='font-family:Nunito;font-weight:700;color:#1a3a5c;margin:0 0 0.8rem'>
                    {emoji} {subject}
                </h4>
            </div>
        """, unsafe_allow_html=True)

        cols = st.columns(len(papers) if len(papers) <= 3 else 3)
        for i, paper in enumerate(papers):
            with cols[i % 3]:
                st.markdown(f"""
                    <a href="{paper['url']}" target="_blank" style="text-decoration:none">
                        <div style='background:#f8faff;border:1.5px solid {color};
                            border-radius:10px;padding:0.7rem 0.9rem;
                            text-align:center;transition:all 0.2s;cursor:pointer;
                            margin-bottom:0.5rem'>
                            <div style='font-size:1.3rem'>📄</div>
                            <div style='font-size:0.75rem;font-weight:700;
                                color:#1a3a5c;margin-top:0.2rem'>
                                {paper['label']}
                            </div>
                            <div style='font-size:0.7rem;color:{color};
                                font-weight:600;margin-top:0.2rem'>
                                ⬇️ Download
                            </div>
                        </div>
                    </a>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background:linear-gradient(135deg,#fef3c7,#fde68a);
            border:2px solid #f59e0b;border-radius:12px;
            padding:1rem 1.2rem;font-size:0.83rem;color:#92400e'>
            💡 <strong>Study Tip:</strong> Always practise past papers under timed conditions.
            Start with the most recent year and work backwards.
            Check your answers against the memorandum also available on the DBE website.
        </div>
    """, unsafe_allow_html=True)


# ==================== ROUTER ====================
def main():
    init_session()
    page = st.session_state.page
    if page == "login":
        page_login()
    elif page == "subject_setup":
        page_subject_setup()
    elif page == "study_guides":
        if not st.session_state.user:
            st.session_state.page = "login"; st.rerun()
        page_study_guides()
    elif page == "previous_papers":
        if not st.session_state.user:
            st.session_state.page = "login"; st.rerun()
        page_previous_papers()
    elif page == "dashboard":
        if not st.session_state.user:
            st.session_state.page = "login"; st.rerun()
        page_dashboard()
    elif page == "subject":
        page_subject()
    elif page == "quiz":
        page_quiz()
    elif page == "tracking":
        page_tracking()
    else:
        st.session_state.page = "login"; st.rerun()

if __name__ == "__main__":
    main()
