# styles/main.py
CSS_STYLES = """
[data-testid="stAppViewContainer"] {
    background-image: url('https://img.freepik.com/free-photo/abstract-digital-grid-black-background_53876-97647.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.about-page [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    background-image: none !important;
}

.main-title { 
    font-size: 2.5rem; 
    font-weight: 700; 
    margin-bottom: 0.5rem; 
    text-align:center; 
    color: #fff; 
    text-shadow: 1px 1px 7px #000; 
}

.sub { 
    color: #e5e7eb; 
    margin-bottom: 2rem; 
    text-align:center; 
    font-weight:500; 
    font-size:1.2rem; 
    text-shadow: 1px 1px 4px #000; 
}

.pred-card { 
    border: 1px solid #e5e7eb; 
    border-radius:15px; 
    padding:20px; 
    background:rgba(255,255,255,0.95); 
    box-shadow:0 8px 16px rgba(0,0,0,0.15); 
    max-width:320px; 
    margin:24px auto; 
    font-weight:600; 
    text-align:center; 
    color:#111827; 
    backdrop-filter: blur(10px);
}

.confidence-high { color: #10b981; font-weight: 700; }
.confidence-medium { color: #f59e0b; font-weight: 700; }
.confidence-low { color: #ef4444; font-weight: 700; }

/* ===== FRONT PAGE BUTTON STYLING ===== */
[data-testid="stButton"] button {
    padding: 12px 28px !important;
    font-size: 16px !important;
    border-radius: 30px !important;
    font-weight: 600 !important;
    font-family: 'Times New Roman', Times, serif !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
    height: auto !important;
    color: white !important;
    background: linear-gradient(135deg, #374151 0%, #1f2937 100%) !important;
    border: 2px solid #4b5563 !important;
}

[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
}

[data-testid="stButton"] button:active {
    background: linear-gradient(135deg, #111827 0%, #000000 100%) !important;
    transform: translateY(0px) !important;
}

.record-btn { 
    background: linear-gradient(45deg, #3b82f6, #1d4ed8); 
    color: white; 
    border: none; 
    padding: 12px 24px; 
    border-radius: 25px; 
    font-weight: 600; 
    font-size: 1.1rem; 
    box-shadow: 0 4px 15px rgba(59,130,246,0.4); 
    width: 100%;
}

.stButton > button:hover {
    transform: scale(1.05);
}

/* ===== TEXT VISIBILITY FOR DARK BACKGROUND ===== */
.stMarkdown {
    color: #ffffff !important;
}

.stMarkdown p {
    color: #e5e7eb !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

.stSubheader {
    color: #ffffff !important;
}

.stTabs [data-baseweb="tab-list"] button {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #00d9ff !important;
    border-bottom-color: #00d9ff !important;
}

label, .stTextInput label, .stSelectbox label, .stSlider label {
    color: #ffffff !important;
    font-weight: 500 !important;
}

.stFileUploader label {
    color: #ffffff !important;
}

p, span, div:not(.pred-card) {
    color: #e5e7eb;
}

.stCaption {
    color: #ffffff !important;
    font-weight: 500 !important;
}

.stAlert p, .stAlert div {
    color: #1f2937 !important;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

.header-container {
    text-align: center;
    margin-top: 60px;
    margin-bottom: 40px;
}

.main-heading {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.6);
}

.sub-heading {
    font-size: 1.25rem;
    font-weight: 500;
    color: #e5e7eb;
    margin-top: 10px;
}

/* ===== CENTER ALL TAB CONTENT ===== */
.stTabs [data-baseweb="tab-panel"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}

/* Center and size subheader inside tabs */
.stTabs h3 {
    text-align: center !important;
    font-size: 1.7rem !important;
    width: 100% !important;
}

/* Center and constrain the info/alert box */
.stAlert {
    max-width: 680px !important;
    margin: 0 auto 16px auto !important;
    font-size: 1.05rem !important;
    text-align: center !important;
}

/* Center the audio input widget */
[data-testid="stAudioInput"] {
    max-width: 680px !important;
    margin: 0 auto !important;
    width: 100% !important;
}

[data-testid="stAudioInput"] label {
    text-align: center !important;
    font-size: 1.15rem !important;
    display: block !important;
    width: 100% !important;
}

[data-testid="stAudioInput"] > div {
    max-width: 680px !important;
    margin: 0 auto !important;
    width: 100% !important;
}

/* Center the file uploader */
[data-testid="stFileUploader"] {
    max-width: 680px !important;
    margin: 0 auto !important;
    width: 100% !important;
}

[data-testid="stFileUploader"] label {
    text-align: center !important;
    font-size: 1.15rem !important;
    display: block !important;
    width: 100% !important;
}

[data-testid="stFileUploader"] > div {
    max-width: 680px !important;
    margin: 0 auto !important;
    width: 100% !important;
}

/* Center footer caption */
.stCaptionContainer,
[data-testid="stCaptionContainer"] {
    text-align: center !important;
    max-width: 680px !important;
    margin: 10px auto !important;
    font-size: 1.0rem !important;
}

/* Center markdown blocks inside tabs */
.stTabs .stMarkdown {
    text-align: center !important;
    max-width: 680px !important;
    margin: 0 auto !important;
}
"""
