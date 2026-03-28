import streamlit as st
import requests

st.set_page_config(page_title="Insurance Predictor", page_icon="💡", layout="wide")

# Custom CSS for a vibrant and modern dynamic design
st.markdown("""
<style>
    body {
        background-color: #0f172a;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-align: center;
        animation: fadeIn 1.5s ease-in-out;
    }
    .prediction-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease;
    }
    .prediction-box:hover {
        transform: translateY(-5px);
    }
    .claim-amount {
        font-size: 3rem;
        font-weight: bold;
        color: #34d399;
        margin-top: 10px;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Insurance Claim Predictor</div>', unsafe_allow_html=True)
st.write("Enter your details below to get an estimated insurance claim prediction.")

# Form Layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=120, value=30, step=1)
    sex = st.selectbox("Sex", ["male", "female"])
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

with col2:
    hereditary_diseases = st.selectbox("Hereditary Diseases", [
        "NoDisease", "Epilepsy", "EyeDisease", "Alzheimer", "Arthritis", "HeartDisease", "Diabetes", "Cancer", "High BP", "Obesity"
    ])
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Smoker", [0, 1])
    city = st.text_input("City", value="NewYork")

with col3:
    bloodpressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    diabetes = st.selectbox("Diabetes", [0, 1])
    regular_ex = st.selectbox("Regular Exercise", [0, 1])
    job_title = st.text_input("Job Title", value="Student")

if st.button("Predict Claim Amount", type="primary", use_container_width=True):
    payload = {
        "age": age,
        "sex": sex,
        "weight": weight,
        "bmi": bmi,
        "hereditary_diseases": hereditary_diseases,
        "no_of_dependents": no_of_dependents,
        "smoker": smoker,
        "city": city,
        "bloodpressure": bloodpressure,
        "diabetes": diabetes,
        "regular_ex": regular_ex,
        "job_title": job_title
    }

    try:
        # Communicate with FastAPI backend inside the same docker network or localhost
        API_URL = "http://localhost:8000/predict"
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            predicted_value = result.get("predicted_claim", 0)
            
            st.markdown(f"""
            <div class="prediction-box">
                <p style="font-size: 1.5rem; color: #94a3b8;">Estimated Claim Amount</p>
                <div class="claim-amount">${predicted_value:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.toast("Prediction successful!", icon="🎉")
            st.balloons()
        else:
            st.error(f"Error from API: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the Backend API. Make sure FastAPI is running on port 8000.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
