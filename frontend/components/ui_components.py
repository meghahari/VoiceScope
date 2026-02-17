import streamlit as st
from styles.main import CSS_STYLES

def load_styles():
    """Load CSS styles"""
    st.markdown(f'<style>{CSS_STYLES}</style>', unsafe_allow_html=True)

def render_header():
    """Render main header"""
    st.markdown('<div class="main-title">🎙️ Voice Age, Gender & Accent Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Resnet With Transfer-Learning Model</div>', unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.header("⚙️ Settings")
    model_path = st.sidebar.text_input("Model path (.keras/.h5)", "my_model.h5")
    record_duration = st.sidebar.slider("Recording duration (seconds)", 3, 15, 7)
    return model_path, record_duration

def render_recording_tab(audio_processor, model):
    """Render live recording tab"""
    st.subheader("🎤 Start Speaking!")
    
    # Global audio storage
    if 'latest_audio' not in st.session_state:
        st.session_state.latest_audio = None
        st.session_state.latest_prediction = None
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎙️ Start Recording", key="record_btn", 
                    help="Click and speak clearly!", type="primary"):
            st.rerun()
    
    # Simplified recording placeholder (WebRTC implementation in main app)
    if st.session_state.latest_audio is not None:
        show_prediction(st.session_state.latest_prediction)
