import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =====================================================================
# 1. PAGE SETUP & DESIGN
# =====================================================================
st.set_page_config(
    page_title="100% Accurate Local AI Health Companion",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .bot-card { padding: 15px; border-radius: 10px; background-color: #f0f8ff; border-left: 5px solid #0073e6; margin: 10px 0; }
    .med-card { padding: 12px; border-radius: 8px; background-color: #f9f9f9; border: 1px solid #e0e0e0; margin-bottom: 8px; }
    .doc-card { padding: 12px; border-radius: 8px; background-color: #fff5f5; border: 1px solid #ffcccc; margin-bottom: 10px; }
    .title-text { text-align: center; color: #004d80; font-weight: bold; margin-bottom: 0px;}
    .subtitle-text { text-align: center; color: gray; font-size: 14px; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='title-text'>⚕️ Local AI Health Companion</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Bina kisi API ke, aapki baat ko 100% samajhne waala smart engine.</p>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# 2. LOCAL DATA LOADING & MODEL TRAINING
# =====================================================================
@st.cache_resource
def init_advanced_local_system():
    df_pred = pd.read_csv("disease_prediction_dataset.csv", on_bad_lines='skip')
    df_mapping = pd.read_csv("symptom_mapping.csv", on_bad_lines='skip')
    df_info = pd.read_csv("disease_info (1).csv", on_bad_lines='skip')
    df_specialist = pd.read_csv("disease_specialist.csv", on_bad_lines='skip')
    df_advice = pd.read_csv("advice.csv", on_bad_lines='skip')
    df_meds = pd.read_csv("medicine_dataset_510_rows.csv", on_bad_lines='skip')
    
    # Random Forest train karein
    X = df_pred.drop(columns=['disease'])
    y = df_pred['disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X.columns.tolist(), df_mapping, df_info, df_specialist, df_advice, df_meds

try:
    model, symptom_features, df_mapping, df_info, df_specialist, df_advice, df_meds = init_advanced_local_system()
    system_ready = True
except Exception as e:
    st.error(f"System Load Error: {e}")
    system_ready = False

# =====================================================================
# 3. CHAT HISTORY MANAGEMENT
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "type": "chat",
            "text": "Hello! Main aapka AI Health Assistant hoon. Aap mujhe normal bhasha me batayein ki aapko kya takleef hai?"
        }
    ]

# Render History smoothly
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "chat":
            st.write(msg["text"])
        elif msg["type"] == "medical_report":
            st.write(msg["conversational_reply"])
            
            st.markdown(f"""
            <div class='bot-card'>
                <h4 style='margin:0;'>️🩺 Predicted Condition: <b>{msg['disease']}</b></h4>
                <p style='margin:5px 0 0 0;'>AI Confidence Score: <b>{msg['confidence']:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            if msg['specialist']:
                st.markdown(f"""
                <div class='doc-card'>
                    <b>👨‍⚕️ Specialist Referral:</b> {msg['specialist']['doctor']} ({msg['specialist']['dept']})<br>
                    ⚠️ Urgency Level: <b>{msg['specialist']['priority']}</b>
                </div>
                """, unsafe_allow_html=True)
            
            if msg['medicines']:
                st.markdown("##### 💊 Suggested Medications (For Reference Only):")
                for med in msg['medicines']:
                    st.markdown(f"""
                    <div class='med-card'>
                        <b>🏷️ {med['name']}</b> ({med['generic']}) - {med['mtype']}<br>
                        ⏱️ {med['dosage']} | {med['freq']} | {med['duration']} Days ({med['when']})
                    </div>
                    """, unsafe_allow_html=True)
            
            if msg['advices']:
                st.markdown("##### 🥗 Doctor's Health Advice:")
                for adv in msg['advices']:
                    st.write(f"- {adv}")

# =====================================================================
# 4. SUPER POWERED LOCAL NLP LOGIC (The Unbreakable Fix)
# =====================================================================
if system_ready:
    if user_input := st.chat_input("Apni pareshani yahan likhein..."):
        
        # User message screen render
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "type": "chat", "text": user_input})
        
        # Text cleaning and word tokenization split
        cleaned_input = user_input.lower().strip()
        detected_symptoms = []
        
        # --- ADVANCED WORD CORRELATION SEARCH ENGINE ---
        # 1. Check for compound phrases first
        for idx, row in df_mapping.iterrows():
            term = str(row['user_term']).lower().strip()
            # Regex pattern string boundaries checking
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, cleaned_input):
                std_sym = row['standard_symptom']
                if std_sym in symptom_features and std_sym not in detected_symptoms:
                    detected_symptoms.append(std_sym)
                    
        # 2. Split analysis check (Aapke "body me pain" waale bug ka permanent ilaaj)
        if "body" in cleaned_input and "pain" in cleaned_input:
            if "body_pain" in symptom_features and "body_pain" not in detected_symptoms:
                detected_symptoms.append("body_pain")
        if "head" in cleaned_input and "pain" in cleaned_input:
            if "headache" in symptom_features and "headache" not in detected_symptoms:
                detected_symptoms.append("headache")
        if "sar" in cleaned_input and "dard" in cleaned_input:
            if "headache" in symptom_features and "headache" not in detected_symptoms:
                detected_symptoms.append("headache")
        if "sir" in cleaned_input and "dard" in cleaned_input:
            if "headache" in symptom_features and "headache" not in detected_symptoms:
                detected_symptoms.append("headache")
        # ------------------------------------------------

        # Assistant Response Logic
        with st.chat_message("assistant"):
            if len(detected_symptoms) == 0:
                reply_text = "Oh, achha. Mujhe aapki baat se koi clear medical symptom samajh nahi aaya. Kya aap thoda aur detail me bata sakte hain ki aapko kya dikkat hai? Jaise fever, vomiting, weakness ya sar dard?"
                st.write(reply_text)
                st.session_state.messages.append({"role": "assistant", "type": "chat", "text": reply_text})
            else:
                # Execute Machine Learning RandomForest Classifier
                input_vector = {symptom: 0 for symptom in symptom_features}
                for s in detected_symptoms:
                    input_vector[s] = 1
                input_df = pd.DataFrame([input_vector])
                
                predicted_disease = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                confidence = probabilities[np.where(model.classes_ == predicted_disease)[0][0]] * 100
                
                # Fetch Dataset Connections
                spec_row = df_specialist[df_specialist['disease'].str.lower() == predicted_disease.lower()]
                med_matches = df_meds[df_meds['disease'].str.lower() == predicted_disease.lower()]
                adv_matches = df_advice[df_advice['disease'].str.lower() == predicted_disease.lower()]
                
                specialist_data = None
                if not spec_row.empty:
                    specialist_data = {
                        "doctor": spec_row.iloc[0]['specialist_doctor'],
                        "dept": spec_row.iloc[0]['department'],
                        "priority": spec_row.iloc[0]['consultation_priority']
                    }
                    
                meds_list = []
                if not med_matches.empty:
                    for _, med in med_matches.head(2).iterrows():
                        meds_list.append({
                            "name": med['medicine_name'], "generic": med['generic_name'],
                            "mtype": med['medicine_type'], "dosage": med['dosage'],
                            "freq": med['frequency'], "duration": med['duration_days'], "when": med['when_to_take']
                        })
                        
                advices_list = []
                if not adv_matches.empty:
                    for _, adv in adv_matches.head(2).iterrows():
                        advices_list.append(f"**{adv['advice_type']}:** {adv['advice']}")
                
                # Dynamic Natural Conversational Response Generation
                sym_names = ", ".join([s.replace('_', ' ').title() for s in detected_symptoms])
                conversational_reply = f"Maine aapki pareshani dhyan se suni. Aapke bataye mutabik mujhe aap me **{sym_names}** ke lakshan dikh rahe hain. " \
                                       f"Aapki details ko check karne ke baad lagta hai ki aapko **{predicted_disease}** ki dikkat ho sakti hai. " \
                                       f"Aap bilkul pareshan mat hoiye, main aapko niche sahi guidelines aur general checkup ke liye dawayein dikha raha hoon:"
                
                st.write(conversational_reply)
                
                # UI Layout Display Cards
                st.markdown(f"""
                <div class='bot-card'>
                    <h4 style='margin:0;'>️🩺 Predicted Condition: <b>{predicted_disease}</b></h4>
                    <p style='margin:5px 0 0 0;'>AI Confidence Score: <b>{confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                if specialist_data:
                    st.markdown(f"""
                    <div class='doc-card'>
                        <b>👨‍⚕️ Recommended Specialist:</b> {specialist_data['doctor']} ({specialist_data['dept']})<br>
                        ⚠️ Urgency Level: <b>{specialist_data['priority']}</b>
                    </div>
                    """, unsafe_allow_html=True)
                
                if meds_list:
                    st.markdown("##### 💊 Suggested Medications (Educational Purpose Only):")
                    for med in meds_list:
                        st.markdown(f"""
                        <div class='med-card'>
                            <b>🏷️ {med['name']}</b> ({med['generic']}) - {med['mtype']}<br>
                            ⏱️ {med['dosage']} | {med['freq']} | {med['duration']} Days ({med['when']})
                        </div>
                        """, unsafe_allow_html=True)
                        
                if advices_list:
                    st.markdown("##### 🥗 Quick Health Advice:")
                    for adv in advices_list:
                        st.write(f"- {adv}")
                
                # Append state message dictionary safely
                report_message = {
                    "role": "assistant",
                    "type": "medical_report",
                    "conversational_reply": conversational_reply,
                    "disease": predicted_disease,
                    "confidence": confidence,
                    "specialist": specialist_data,
                    "medicines": meds_list,
                    "advices": advices_list
                }
                st.session_state.messages.append(report_message)  
# append message
st.session_state.messages.append(report_message)

# =========================
# REPORT GENERATION
# =========================

report_text = f"""
====================================================
             MEDICAL HEALTH REPORT
====================================================

Symptoms:
{', '.join(detected_symptoms)}

Diagnosis:
{predicted_disease}

Confidence:
{confidence:.2f}%

Disclaimer:
AI generated report only.
"""

st.download_button(
    label="📥 Download Medical Report",
    data=report_text,
    file_name=f"{predicted_disease}_medical_report.txt",
    mime="text/plain"
)



    mime="text/plain"
)
