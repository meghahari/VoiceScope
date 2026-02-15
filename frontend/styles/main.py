# styles/main.py
CSS_STYLES = """
[data-testid="stAppViewContainer"] {
    background-image: url('https://img.freepik.com/free-photo/abstract-digital-grid-black-background_53876-97647.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* About Page Styling */
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
    color: #d1d5db; 
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
    padding: 10px 24px !important;
    font-size: 0.95rem !important;
    border-radius: 25px !important;
    font-weight: 600 !important;
    font-family: 'Times New Roman', Times, serif !important;
    transition: all 0.3s ease !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    height: auto !important;
}

/* Get Started Button (Primary) */
button[key="get_started"] {
    background: linear-gradient(135deg, #00d9ff 0%, #00a8cc 100%) !important;
    color: white !important;
}

button[key="get_started"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4) !important;
}

/* Learn More Button (Secondary) */
button[key="learn_more"] {
    background: linear-gradient(135deg, #9d4edd 0%, #7b2cbf 100%) !important;
    color: white !important;
}

button[key="learn_more"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(157, 78, 221, 0.4) !important;
}

/* Recording Button */
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

/* ===== FRONT PAGE / MAIN HEADER ===== */
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
"""
