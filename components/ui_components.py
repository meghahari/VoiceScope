import streamlit as st
from styles.main import CSS_STYLES

def load_styles():
    """Load CSS styles"""
    st.markdown(f'<style>{CSS_STYLES}</style>', unsafe_allow_html=True)
    
    # Additional inline styles for white text on dark background
    st.markdown(
        """
        <style>
        /* Force white text on main page */
        .main .block-container {
            color: #ffffff !important;
        }
        
        /* All text elements white */
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
            color: #ffffff !important;
        }
        
        /* Headings white + Times New Roman */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-family: 'Times New Roman', Times, serif !important;
        }

        /* Main title Times New Roman */
        .main-title {
            font-family: 'Times New Roman', Times, serif !important;
        }
        
        /* Tab labels - bright colors */
        .stTabs [data-baseweb="tab-list"] button {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #00d9ff !important;
            border-bottom-color: #00d9ff !important;
        }
        
        /* Labels white */
        label {
            color: #ffffff !important;
        }
        
        /* Caption white */
        .stCaption {
            color: #ffffff !important;
        }
        
        /* Info boxes keep their background colors but white text */
        .stAlert {
            color: #1f2937 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_header():
    """Render main header - WITHOUT subtitle"""
    st.markdown(
        '<div class="main-title" style="color:#ffffff !important; font-family: \'Times New Roman\', Times, serif !important; font-size: 2.5rem; font-weight: bold; text-align: center; padding: 1rem 0;">🎙️ Voice Age, Gender & Accent Classification</div>', 
        unsafe_allow_html=True
    )

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
