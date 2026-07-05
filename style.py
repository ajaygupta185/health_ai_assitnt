import streamlit as st
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import base64

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM CSS INJECTION
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #F4F6F9;
    }
    [data-testid="stSidebar"] {
        background-color: #0F1E36 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .sidebar-header {
        text-align: center;
        padding: 20px 10px;
    }
    .sidebar-header h2 {
        font-size: 22px;
        margin-top: 10px;
    }
    .bot-bubble {
        background-color: #EBF3FF;
        color: #1C2D42;
        padding: 15px;
        border-radius: 12px 12px 12px 0px;
        margin-bottom: 5px;
        border: 1px solid #D6E4FA;
        font-family: sans-serif;
    }
    .user-bubble {
        background-color: #E2F5E1;
        color: #1A3E1C;
        padding: 15px;
        border-radius: 12px 12px 0px 12px;
        margin-bottom: 5px;
        border: 1px solid #C8E6C9;
        font-family: sans-serif;
    }
    .chat-time {
        font-size: 11px;
        color: #7A8B9E;
        margin-bottom: 15px;
    }
    .download-btn {
        background-color: #1A56C6;
        color: white !important;
        padding: 10px 20px;
        border-radius: 6px;
        text-decoration: none;
        display: inline-block;
        text-align: center;
        font-weight: 500;
        margin-top: 10px;
    }
    .prediction-container {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #EAECEF;
    }
    .disease-title {
        color: #D9383A !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        margin-top: 0px !important;
        margin-bottom: 15px !important;
    }
    .confidence-text {
        color: #2E7D32 !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        margin-top: 0px !important;
        margin-bottom: 15px !important;
    }
    .card-label {
        font-size: 13px !important;
        color: #7A8B9E !important;
        margin-bottom: 2px !important;
        font-weight: 500;
    }
    stToolbar, [data-testid="stToolbar"], [data-testid="stStatusWidget"] {
        display: none !important;
    }
    header {
        visibility: hidden !important;
        height: 0px !important;
    }
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MULTI-LANGUAGE BACKEND RESOURCE LOADER
# ==========================================
@st.cache_resource
def load_backend_resources():
    try:
        model = joblib.load("health_model.joblib")
        med_df = pd.read_csv("medicine_dataset_510_rows.csv")
        advice_df = pd.read_csv("advice.csv")
        
        # 🌟 Loading your Symptom Mapping File
        symptom_map_df = pd.read_csv("symptom_mapping.csv")
        # Longest phrases matched first to avoid substring cutting conflicts
        symptom_map_df['term_len'] = symptom_map_df['user_term'].str.len()
        symptom_map_df = symptom_map_df.sort_values(by='term_len', ascending=False)
        
        return model, med_df, advice_df, symptom_map_df
    except Exception as e:
        return None, None, None, None

model_pipeline, med_df, advice_df, symptom_map_df = load_backend_resources()

# ==========================================
# 4. PREDICTION ENGINE (USES YOUR CSV FILE)
# ==========================================
def get_local_prediction(user_text):
    current_time = datetime.now().strftime("%I:%M %p | %d %b %Y")
    text = user_text.lower().strip()
    
    if model_pipeline is None or symptom_map_df is None:
        return {
            "disease": "System Loading", "confidence": "0%", "medicine": "None",
            "advice": "Please verify backend files are ready.", 
            "bot_response": "Maaf kijiyega babu, files loading me dikkat ba.", "time": current_time
        }
    
    # 51 exact model symptoms
    symptom_features = [
        'fever', 'high_fever', 'headache', 'migraine_pain', 'body_pain', 'joint_pain', 'muscle_pain', 
        'fatigue', 'weakness', 'dizziness', 'cough', 'dry_cough', 'productive_cough', 'sore_throat', 
        'runny_nose', 'sneezing', 'breathing_difficulty', 'chest_pain', 'palpitations', 'vomiting', 
        'nausea', 'diarrhea', 'constipation', 'stomach_pain', 'abdominal_pain', 'loss_of_appetite', 
        'weight_loss', 'weight_gain', 'night_sweats', 'chills', 'dehydration', 'skin_rash', 'itching', 
        'red_eyes', 'blurred_vision', 'hearing_loss', 'ear_pain', 'swelling', 'anxiety', 'depression', 
        'insomnia', 'memory_loss', 'confusion', 'tremor', 'seizure', 'frequent_urination', 
        'burning_urination', 'blood_in_urine', 'yellow_skin', 'hair_loss', 'bone_pain'
    ]
    
    # Extract symptoms directly from your 5,100 row database mapping
    input_vector = [0] * len(symptom_features)
    any_symptom_detected = False
    
    for _, row in symptom_map_df.iterrows():
        term = str(row['user_term']).lower().strip()
        if term in text:
            std_symptom = row['standard_symptom']
            
            # Align synonyms variant names with trained features
            if std_symptom == 'migraine':
                std_symptom = 'migraine_pain'
            elif std_symptom == 'back_pain':
                std_symptom = 'body_pain'
            elif std_symptom == 'throat_pain':
                std_symptom = 'sore_throat'
            elif std_symptom == 'cold':
                std_symptom = 'runny_nose'
            
            if std_symptom in symptom_features:
                idx = symptom_features.index(std_symptom)
                input_vector[idx] = 1
                any_symptom_detected = True

    # Fail-safe mechanism for unclear expressions
    if not any_symptom_detected:
        return {
            "disease": "Awaiting Input", "confidence": "0%", "medicine": "None",
            "advice": "Please mention specific symptoms like bukhar, sir dard, loose motion.",
            "bot_response": "Babu, hum tihar baat thoda samajh nahi paye. Kripya batiye ki aapko bukhar, pet dard, loose motion ya sir dard me se kya dikkat ba?",
            "time": current_time
        }
        
    input_df = pd.DataFrame([input_vector], columns=symptom_features)
    predicted_disease = model_pipeline.predict(input_df)[0]
    probabilities = model_pipeline.predict_proba(input_df)[0]
    class_idx = np.where(model_pipeline.classes_ == predicted_disease)[0][0]
    confidence_score = int(probabilities[class_idx] * 100)
    
   # 💊 UPDATED MEDICINE DATABASE LOOKUP
    try:
        clean_predicted = predicted_disease.strip().lower()
        # Ensure 'disease' column exists and is string
        med_match = med_df[med_df['disease'].astype(str).str.strip().str.lower() == clean_predicted]
        
        if not med_match.empty:
            row = med_match.iloc[0]
            recommended_medicine = f"{row['medicine_name']} ({row['dosage']})"
        else:
            # Natural Bhojpuri fallback jab dawa na mile
            recommended_medicine = "kisi acche doctor se milke sahi dawa likhwaiye"
    except Exception as e:
        recommended_medicine = "kisi acche doctor se milke sahi dawa likhwaiye"

    # Bot response logic (Isse aapka chat bubble natural dikhega)
    if "doctor" in recommended_medicine:
        bot_response = f"Babu, tihar bataye hue lakshano se <b>{predicted_disease}</b> ke dikkat ho sake la. Ee thoda gambhir bimari ho sake la, isliye hum dawa nahi bata rahe. Kripya <b>{recommended_medicine}</b>."
    else:
        bot_response = f"Babu, tihar bataye hue lakshano ke hisab se ee <b>{predicted_disease}</b> ke dikkat ho sake la. Bilkul chinta mat kara. Abhi ke liye aap <b>{recommended_medicine}</b> le lijiye aur thoda aaram kariye."
        
    # Advice lookup
    try:
        advice_match = advice_df[advice_df['disease'].str.lower() == predicted_disease.lower()]
        advice_text = advice_match['advice'].values[0] if not advice_match.empty else "Thoda aaram kariye aur paani piyat rahi."
    except:
        advice_text = "Thoda aaram kariye."

    bot_response = f"Babu, tihar bataye hue lakshano ke hisab se ee <b>{predicted_disease}</b> ke dikkat ho sake la. Bilkul chinta mat kara. Abhi ke liye aap ek <b>{recommended_medicine}</b> le lijiye, thoda aaram kariye."

    return {
        "disease": predicted_disease, "confidence": f"{confidence_score}%", "medicine": recommended_medicine,
        "advice": advice_text, "bot_response": bot_response, "time": current_time
    }

# ==========================================
# 5. INITIALIZE SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "bot",
            "text": "Ka babu ka haal ba? Thik baran nu?<br>Batao, aaj ee tihar bhaee tor ka seva kar sake la?<br><i>(Kya pareshani hai aapko?)</i>",
            "time": "10:30 AM"
        }
    ]

if "prediction" not in st.session_state:
    st.session_state.prediction = {
        "disease": "Awaiting Input", "confidence": "0%", "medicine": "None",
        "advice": "Please type your symptoms to start diagnosis.", "time": "--:-- | -- --- ----"
    }

# ==========================================
# 6. SIDEBAR IMPLEMENTATION
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <span style="font-size: 50px; color: #E94560;">❤️</span>
        <h2>AI Health Assistant</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("💬 New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = [
            {
                "role": "bot",
                "text": "Ka babu ka haal ba? Thik baran nu?<br>Batao, aaj ee tihar bhaee tor ka seva kar sake la?<br><i>(Kya pareshani hai aapko?)</i>",
                "time": "10:30 AM"
            }
        ]
        st.session_state.prediction = {
            "disease": "Awaiting Input", "confidence": "0%", "medicine": "None",
            "advice": "Please type your symptoms to start diagnosis.", "time": "--:-- | -- --- ----"
        }
        st.rerun()
    
    if st.button("🕒 Chat History", use_container_width=True):
        st.toast("Chat History feature jaldi hi chalu hoga! ⏳")
    if st.button("❤️ Health Tips", use_container_width=True):
        st.info("💡 **Health Tip:** Roz kam se kam 8-10 gilaas paani pijiye aur screen exposure kam kariye!")
    if st.button("ℹ️ About Project", use_container_width=True):
        st.success("🤖 **AI Health Assistant:** Multi-lingual dynamic matching console.")
    
    st.write("---")
    st.markdown("""
    <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #FFA500;">
        <h5 style="color: #FFA500 !important; margin: 0 0 5px 0;">Disclaimer</h5>
        <p style="font-size: 12px; color: #BAC4D1 !important; line-height: 1.4;">
            This AI Health Assistant is for educational purposes only. It is not a substitute for professional medical advice, diagnosis or treatment.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 7. MAIN PAGE HEADER
# ==========================================
head_col1, head_col2 = st.columns([3, 1])

with head_col1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px;">
        <span style="font-size: 35px; color: #E94560;">❤️</span>
        <div>
            <h2 style="margin:0; padding:0; color: #0F1E36;">AI Health Assistant</h2>
            <p style="margin:0; color: #6C7A89; font-size:14px;">Intent-Driven Conversational Health Assistant</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with head_col2:
    pred_data = st.session_state["prediction"]
    prescription_text = f"""==================================================
  AI HEALTH ASSISTANT - OFFICIAL E-PRESCRIPTION
==================================================
Report Generated On : {pred_data['time']}
--------------------------------------------------
[DIAGNOSIS REPORT]
Predicted Disease   : {pred_data['disease']}
Confidence Score    : {pred_data['confidence']}

[TREATMENT PLAN]
Recommended Medicine: {pred_data['medicine']}

[ADVICE & PRECAUTIONS]
{pred_data['advice']}
==================================================
"""
    b64_content = base64.b64encode(prescription_text.encode()).decode()
    download_url = f"data:text/plain;base64,{b64_content}"
    st.markdown(f'<div style="text-align: right;"><a href="{download_url}" download="AI_E_Prescription.txt" class="download-btn" style="font-size: 14px; padding: 6px 12px;">📥 Download Prescription</a></div>', unsafe_allow_html=True)

st.write("---")

# ==========================================
# 8. BODY CONTENT (CHAT FEED SYSTEM)
# ==========================================
body_col1, body_col2 = st.columns([2.2, 1])

with body_col1:
    def handle_submit():
        user_input = st.session_state.user_text_input
        if user_input.strip() != "":
            now = datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({"role": "user", "text": user_input, "time": f"{now} ✔️"})
            
            # Predict from dynamic CSV mapping
            result = get_local_prediction(user_input)
            st.session_state.messages.append({"role": "bot", "text": result["bot_response"], "time": now})
            
            # Create live prescription string download link inside bubble
            pred_data_live = result
            prescription_text_live = f"""==================================================
  AI HEALTH ASSISTANT - OFFICIAL E-PRESCRIPTION
==================================================
Report Generated On : {pred_data_live['time']}
--------------------------------------------------
[DIAGNOSIS REPORT]
Predicted Disease   : {pred_data_live['disease']}
Confidence Score    : {pred_data_live['confidence']}

[TREATMENT PLAN]
Recommended Medicine: {pred_data_live['medicine']}

[ADVICE & PRECAUTIONS]
{pred_data_live['advice']}
==================================================
"""
            b64_content_live = base64.b64encode(prescription_text_live.encode()).decode()
            download_url_live = f"data:text/plain;base64,{b64_content_live}"
            
            st.session_state.messages.append({
                "role": "bot",
                "text": "Kya aap is bimari aur dawa ka official E-Prescription download karna chahte hain? Niche diye gaye button par click karein.<br><br><a href='" + download_url_live + "' download='AI_E_Prescription.txt' class='download-btn'>📥 Download Official E-Prescription PDF</a>",
                "time": now
            })
            
            st.session_state.prediction = {
                "disease": result["disease"], "confidence": result["confidence"],
                "medicine": result["medicine"], "advice": result["advice"], "time": result["time"]
            }
            st.session_state.user_text_input = ""

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "bot":
                st.markdown(f'<div class="bot-bubble"><strong>🤖 Assistant</strong><br>{msg["text"]}</div><div class="chat-time">{msg["time"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end;"><div class="user-bubble" style="text-align: left;"><strong>👤 You</strong><br>{msg["text"]}</div><div class="chat-time">{msg["time"]}</div></div>', unsafe_allow_html=True)

    st.write("")
    input_col1, input_col2 = st.columns([5, 1])
    with input_col1:
        st.text_input("Type your symptoms here...", label_visibility="collapsed", placeholder="Type your symptoms here...", key="user_text_input", on_change=handle_submit)
    with input_col2:
        st.button("✈️ Send", use_container_width=True, on_click=handle_submit)

# Right Side Panel
predicted_disease = st.session_state.prediction["disease"]
confidence_score = st.session_state.prediction["confidence"]
recommended_medicine = st.session_state.prediction["medicine"]
advice_text = st.session_state.prediction["advice"]
prediction_time = st.session_state.prediction["time"]

with body_col2:
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown('<h4 style="margin-top: 0; color: #0F1E36; border-bottom: 1px solid #EAECEF; padding-bottom: 10px;">Prediction Summary</h4>', unsafe_allow_html=True)
    
    st.markdown('<p class="card-label">Predicted Disease</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="disease-title">{predicted_disease}</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="card-label">Confidence Score</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="confidence-text">{confidence_score}</p>', unsafe_allow_html=True)
    
    st.markdown('<hr style="border: 0; border-top: 1px solid #EAECEF; margin: 15px 0;">', unsafe_allow_html=True)
    
    st.markdown('<p class="card-label">Recommended Medicine</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-weight: bold; color: #0F1E36; margin-bottom: 15px;">{recommended_medicine}</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="card-label">Advice</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #0F1E36; font-size: 14px; line-height: 1.4; margin-bottom: 20px;">{advice_text}</p>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="background-color: #EBF7ED; padding: 10px; border-radius: 8px; text-align: center;"><span style="font-size: 11px; color: #1E4620; font-weight: 600; display: block; text-transform: uppercase; letter-spacing: 0.5px;">Prediction Time</span><span style="font-size: 13px; color: #1E4620; font-weight: bold;">{prediction_time}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)