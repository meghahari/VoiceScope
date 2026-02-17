import streamlit as st

def show_prediction(prediction: tuple):
    """Display age, gender, and accent prediction cards"""
    age_label, gender_label, accent_label = prediction
    
    # Add custom styling for prediction cards
    st.markdown("""
        <style>
        .prediction-header {
            text-align: center;
            color: #ffffff;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 30px;
            font-family: 'Times New Roman', Times, serif;
        }
        
        .pred-card-new {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
            text-align: center;
            transition: transform 0.3s ease;
            margin: 10px;
            backdrop-filter: blur(10px);
        }
        
        .pred-card-new:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        }
        
        .pred-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .pred-title {
            color: #2d3748;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 15px;
            font-family: 'Times New Roman', Times, serif;
        }
        
        .pred-value {
            color: #1a202c;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
            font-family: 'Times New Roman', Times, serif;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Prediction header
    st.markdown('<div class="prediction-header">🎯 Analysis Results</div>', unsafe_allow_html=True)
    
    # Create three columns for predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
            <div class="pred-card-new">
                <div class="pred-icon">👥</div>
                <div class="pred-title">Age Prediction</div>
                <div class="pred-value">{age_label.title()}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
            <div class="pred-card-new">
                <div class="pred-icon">👤</div>
                <div class="pred-title">Gender Prediction</div>
                <div class="pred-value">{gender_label.title()}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
            <div class="pred-card-new">
                <div class="pred-icon">🌍</div>
                <div class="pred-title">Accent Prediction</div>
                <div class="pred-value">{accent_label.title()}</div>
            </div>
        ''', unsafe_allow_html=True)
