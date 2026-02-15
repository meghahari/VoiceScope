import streamlit as st

def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence score"""
    if confidence > 0.85:
        return "confidence-high"
    elif confidence > 0.65:
        return "confidence-medium"
    else:
        return "confidence-low"

def show_prediction(prediction: tuple):
    """Display age and gender prediction cards"""
    age_label, age_conf, gender_label, gender_conf = prediction
    
    # Add custom styling for prediction cards
    st.markdown("""
        <style>
        .prediction-container {
            margin-top: 40px;
            padding: 20px;
        }
        
        .prediction-header {
            text-align: center;
            color: #ffffff;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 30px;
            font-family: 'Times New Roman', Times, serif;
        }
        
        .pred-card-new {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            text-align: center;
            transition: transform 0.3s ease;
            margin: 10px;
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
        
        .pred-confidence {
            color: #4a5568;
            font-size: 16px;
            font-weight: 500;
            font-family: 'Times New Roman', Times, serif;
        }
        
        .confidence-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            margin-top: 10px;
        }
        
        .confidence-high {
            background-color: #d4edda;
            color: #155724;
        }
        
        .confidence-medium {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .confidence-low {
            background-color: #f8d7da;
            color: #721c24;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Prediction header
    st.markdown('<div class="prediction-header">🎯 Analysis Results</div>', unsafe_allow_html=True)
    
    # Create two columns for predictions
    col1, col2 = st.columns(2)
    
    with col1:
        age_conf_class = get_confidence_class(age_conf)
        confidence_label = "High" if age_conf > 0.85 else "Medium" if age_conf > 0.65 else "Low"
        
        st.markdown(f'''
            <div class="pred-card-new">
                <div class="pred-icon">👥</div>
                <div class="pred-title">Age Prediction</div>
                <div class="pred-value">{age_label.title()}</div>
                <div class="pred-confidence">
                    Confidence: {age_conf:.1%}
                </div>
                <div class="confidence-badge {age_conf_class}">
                    {confidence_label} Confidence
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        gender_conf_class = get_confidence_class(gender_conf)
        confidence_label = "High" if gender_conf > 0.85 else "Medium" if gender_conf > 0.65 else "Low"
        
        st.markdown(f'''
            <div class="pred-card-new">
                <div class="pred-icon">👤</div>
                <div class="pred-title">Gender Prediction</div>
                <div class="pred-value">{gender_label.title()}</div>
                <div class="pred-confidence">
                    Confidence: {gender_conf:.1%}
                </div>
                <div class="confidence-badge {gender_conf_class}">
                    {confidence_label} Confidence
                </div>
            </div>
        ''', unsafe_allow_html=True)
