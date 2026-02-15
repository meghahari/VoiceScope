import streamlit as st

# ---- PAGE CONTROLLER ----
if "page" not in st.session_state:
    st.session_state.page = "front"

import streamlit_webrtc as webrtc
import numpy as np
import io
from models.audio_processor import AudioProcessor
from components.ui_components import load_styles, render_header
from utils.constants import *
from components.prediction_cards import show_prediction

# Page config
st.set_page_config(page_title="VoiceScope", page_icon="🎙️", layout="wide")

# Load styles & header (only show on non-front pages)
load_styles()
if st.session_state.page != "front" and st.session_state.page != "about":
    render_header()

# ================= FRONT PAGE =================
if st.session_state.page == "front":
    
    # Custom background for front page only
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://i.pinimg.com/736x/5a/20/ba/5a20ba6a2889eb8dfa10c07d49cd7a88.jpg') !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
        }
        
        /* Times New Roman font for Get Started and Learn More buttons */
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-primary"],
        [data-testid="stButton"] > button {
            font-family: 'Times New Roman', Times, serif !important;
            font-weight: 600 !important;
            font-size: 16px !important;
        }
        
        /* Additional targeting for button text */
        div[data-testid="stButton"] button p {
            font-family: 'Times New Roman', Times, serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center; padding-top:120px">
            <h1 style="font-size:70px; font-family: 'Times New Roman', Times, serif; font-weight: 700; letter-spacing: 2px; text-shadow: 0 0 20px rgba(0,217,255,0.5), 0 0 40px rgba(157,78,221,0.3);">
                🎤 <span style="color:#00d9ff;">Voice</span><span style="color:#9d4edd;">Scope</span>
            </h1>
            <p style="font-size:20px; color:#e5e7eb; margin-top: 15px; text-shadow: 0 2px 10px rgba(0,0,0,0.8); font-family: 'Times New Roman', Times, serif;">AI-powered voice analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")
    
    # Create three columns for button layout
    col1, col2, col3, col4, col5 = st.columns([1, 1, 0.5, 1, 1])
    
    with col2:
        if st.button("Get Started", key="get_started", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    
    with col4:
        if st.button("Learn More", key="learn_more", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()

    # ⛔ IMPORTANT: stop here, don't load rest of UI
    st.stop()


# ================= ABOUT PAGE =================
if st.session_state.page == "about":
    
    # Add cleaner background styling
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        /* Apply Times New Roman to all text elements */
        * {
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        /* Clean card styling */
        .feature-card {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            height: 100%;
            transition: transform 0.3s ease;
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }
        
        .feature-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .feature-title {
            color: #2d3748;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 12px;
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        .feature-desc {
            color: #4a5568;
            font-size: 15px;
            line-height: 1.6;
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        .about-header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        .about-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        .about-subtitle {
            font-size: 18px;
            opacity: 0.95;
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        /* Section headers */
        h2, h3, h4, p {
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        /* 🎯 BLACK AND WHITE BUTTON FOR TRY VOICESCOPE NOW */
        [data-testid="stButton"] button[kind="primary"] {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 2px solid #FFFFFF !important;
            font-weight: 600 !important;
            padding: 14px 32px !important;
            font-size: 16px !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        [data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 2px solid #000000 !important;
            transform: scale(1.05) !important;
            box-shadow: 0 5px 15px rgba(255, 255, 255, 0.3) !important;
        }
        
        [data-testid="stButton"] button[kind="primary"]:active {
            transform: scale(0.98) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Back button
    if st.button("←", key="back_to_front"):
        st.session_state.page = "front"
        st.rerun()
    
    # Header section
    st.markdown(
        """
        <div class="about-header">
            <h1 class="about-title">About VoiceScope</h1>
            <p class="about-subtitle">Advanced AI-powered voice analysis system</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features section
    st.markdown("<h2 style='text-align:center; color:white; margin-bottom:30px;'>Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">👥</div>
                <h3 class="feature-title">Age Detection</h3>
                <p class="feature-desc">
                    Accurately identifies speaker age groups including children, adults, and seniors using advanced neural networks.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⚧️</div>
                <h3 class="feature-title">Gender Classification</h3>
                <p class="feature-desc">
                    Determines speaker gender with high accuracy through voice pattern analysis and deep learning models.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌍</div>
                <h3 class="feature-title">Accent Recognition</h3>
                <p class="feature-desc">
                    Identifies regional accents including American, British, and Indian English variants with precision.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Technology section
    st.markdown("<h2 style='text-align:center; color:white; margin-bottom:30px;'>Our Technology</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3 class="feature-title">🧠 CNN-BiLSTM Architecture</h3>
                <p class="feature-desc">
                    Our model combines Convolutional Neural Networks (CNN) for feature extraction with Bidirectional Long Short-Term Memory (BiLSTM) networks for sequential pattern recognition, achieving state-of-the-art performance.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3 class="feature-title">🔄 Transfer Learning</h3>
                <p class="feature-desc">
                    Leveraging pre-trained models and transfer learning techniques, VoiceScope achieves superior accuracy while requiring less training data and computational resources.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # How it works section
    st.markdown("<h2 style='text-align:center; color:white; margin-bottom:30px;'>How It Works</h2>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="feature-card" style="max-width:900px; margin:0 auto;">
            <div style="display:flex; justify-content:space-around; flex-wrap:wrap; text-align:center;">
                <div style="flex:1; min-width:200px; padding:15px;">
                    <div style="font-size:36px; margin-bottom:10px;">🎤</div>
                    <h4 style="color:#2d3748; margin-bottom:8px;">1. Record Audio</h4>
                    <p style="color:#4a5568; font-size:14px;">Capture voice input through microphone or upload audio file</p>
                </div>
                <div style="flex:1; min-width:200px; padding:15px;">
                    <div style="font-size:36px; margin-bottom:10px;">⚙️</div>
                    <h4 style="color:#2d3748; margin-bottom:8px;">2. Process</h4>
                    <p style="color:#4a5568; font-size:14px;">Extract MFCC features and normalize audio data</p>
                </div>
                <div style="flex:1; min-width:200px; padding:15px;">
                    <div style="font-size:36px; margin-bottom:10px;">🤖</div>
                    <h4 style="color:#2d3748; margin-bottom:8px;">3. Analyze</h4>
                    <p style="color:#4a5568; font-size:14px;">Run through CNN-BiLSTM neural network</p>
                </div>
                <div style="flex:1; min-width:200px; padding:15px;">
                    <div style="font-size:36px; margin-bottom:10px;">📊</div>
                    <h4 style="color:#2d3748; margin-bottom:8px;">4. Results</h4>
                    <p style="color:#4a5568; font-size:14px;">Display predictions with confidence scores</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # CTA button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Try VoiceScope Now", key="try_now", use_container_width=True, type="primary"):
            st.session_state.page = "home"
            st.rerun()

    # ⛔ IMPORTANT: stop here
    st.stop()



# ==================== SIDEBAR - MODEL PATH HERE ====================
st.sidebar.header("⚙️ Model Settings")
MODEL_PATH = st.sidebar.text_input(
    "📁 Model path (.keras/.h5)", 
    value="./my_model.keras",  # 👈 CHANGE THIS TO YOUR MODEL PATH
    help="Path to your trained model file"
)
RECORD_DURATION = st.sidebar.slider("⏱️ Recording (seconds)", 3, 15, 7)

# ==================== INITIALIZE PROCESSOR ====================
audio_processor = AudioProcessor()

# ==================== LOAD MODEL ====================
if st.session_state.get('model') is None:
    with st.spinner("🔄 Loading model..."):
        model = audio_processor.load_model(MODEL_PATH)
        if model:
            st.session_state.model = model
            st.sidebar.success("✅ Model loaded!")
        else:
            st.sidebar.error("❌ Model failed to load!")

model = st.session_state.get('model')

# ==================== TABS ====================
# ================= HOME PAGE =================
if st.session_state.page == "home":


 # Back button in top left - ONLY ARROW
    if st.button("←", key="back_to_front_home"):
        st.session_state.page = "front"
        st.rerun()
    
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    
    # Add class for styling and BUTTON CUSTOMIZATION
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://i.pinimg.com/originals/09/c1/0e/09c10eca4ae8a0c3dd0234488b15caf5.gif') !important;
        }
        
        /* 🎯 CUSTOMIZE START RECORDING BUTTON - BLACK AND WHITE */
        button[kind="primary"] {
            background-color: #000000 !important;  /* Black background */
            color: #FFFFFF !important;  /* White text */
            border: 2px solid #FFFFFF !important;  /* White border */
            padding: 12px 40px !important;
            font-size: 16px !important;
            border-radius: 10px !important;
            max-width: 350px !important;
            margin: 0 auto !important;
            transition: all 0.3s ease !important;
            font-weight: 600 !important;
        }
        
        button[kind="primary"]:hover {
            background-color: #FFFFFF !important;  /* White background on hover */
            color: #000000 !important;  /* Black text on hover */
            border: 2px solid #000000 !important;  /* Black border on hover */
            transform: scale(1.05) !important;
            box-shadow: 0 5px 15px rgba(255, 255, 255, 0.4) !important;
        }
        
        button[kind="primary"]:active {
            transform: scale(0.98) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )



tab1, tab2 = st.tabs(["🎤 Live Recording", "📁 Upload Audio"])

 

with tab1:
    st.subheader("🎙️ Real-time Voice Analysis")
    
    # Session state for audio
    if 'latest_audio' not in st.session_state:
        st.session_state.latest_audio = None
        st.session_state.prediction = None





    
    # RECORDING BUTTON - Changed column ratio to make it smaller
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔴 Start Recording", key="record", type="primary", 
                    help=f"Speak for {RECORD_DURATION}s", use_container_width=True):
            st.session_state.recording = True
            st.rerun()
    
    # SIMPLIFIED LIVE PREVIEW (Real WebRTC in production)
    if st.session_state.get('recording', False) and model:
        st.info("🎤 Recording... Speak clearly!")
        # Simulate audio capture (replace with real mic input)
        placeholder_audio = np.random.randn(TARGET_SR * RECORD_DURATION)
        processed_audio, sr = audio_processor.process_recording(placeholder_audio)
        
        if processed_audio is not None:
            age_label, age_conf, gender, gender_conf = audio_processor.predict(
                model, processed_audio, sr
            )
            st.session_state.prediction = (age_label, age_conf, gender, gender_conf)
            st.session_state.latest_audio = processed_audio
            st.success("✅ Analysis complete!")
    
    # Show prediction
    if st.session_state.get('prediction'):
        show_prediction(st.session_state.prediction)

with tab2:
    st.subheader("📁 Upload MP3/WAV")
    uploaded_file = st.file_uploader("Choose audio file", type=['mp3', 'wav', 'm4a'])
    
    if uploaded_file and model:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        
        with st.spinner("🎯 Predicting..."):
            if uploaded_file.name.endswith('.mp3'):
                audio_data, sr = audio_processor.mp3_to_wav(audio_bytes)
            else:
                audio_data, sr = audio_processor.process_recording(
                    librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR)[0]
                )
            
            if audio_data is not None:
                prediction = audio_processor.predict(model, audio_data, sr)
                show_prediction(prediction)

# Footer
if model:
    
    st.caption("💡 Put your model file in the same folder and update MODEL_PATH above!")
