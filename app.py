import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# ----------------------------------
# Custom CSS (BLACK FONTS + CLASSIC UI)
# ----------------------------------
st.markdown("""
<style>

/* App background */
.stApp {
    background-color: white;
    font-family: 'Georgia', serif;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #f8f1e7;
    border-right: 3px solid #6a1b1a;
}

/* ALL labels & text BLACK */
label, p, span, div {
    color: black !important;
    font-weight: 500;
}

/* Sidebar slider labels */
section[data-testid="stSidebar"] label {
    color: black !important;
    font-weight: 600;
}

/* Headings BLACK */
h1, h2, h3, h4 {
    color: black !important;
}

/* Title styling */
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #6a1b1a;
}

.sub-title {
    text-align: center;
    font-size: 16px;
    color: black;
}

/* Predict button */
.stButton>button {
    background-color: #6a1b1a;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 8px 20px;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #6a1b1a, #b8860b);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    font-size: 26px;
    font-weight: bold;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.3);
    animation: glow 1.5s infinite alternate;
}

/* Glow animation */
@keyframes glow {
    from { box-shadow: 0px 0px 10px #b8860b; }
    to { box-shadow: 0px 0px 25px #ffd700; }
}

/* Footer */
.footer {
    text-align: center;
    font-size: 12px;
    color: black;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Load Model & Scaler
# ----------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("finalized_RFmodel.sav", "rb"))
    scaler = pickle.load(open("scaler_model.csv", "rb"))
    return model, scaler

model, scaler = load_model()

# ----------------------------------
# Title Section
# ----------------------------------
st.markdown("<div class='main-title'>🍷 Wine Quality Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predict the quality of red wine using Machine Learning</div>", unsafe_allow_html=True)

st.divider()

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("Wine Chemical Properties")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.4)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.70)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.00)
residual_sugar = st.sidebar.slider("Residual Sugar (log)", 0.1, 2.0, 0.65)
chlorides = st.sidebar.slider("Chlorides (log)", 0.1, 1.5, 0.90)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1, 70, 20)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide (log)", 5, 200, 98)
density = st.sidebar.slider("Density", 0.990, 1.005, 1.000)
pH = st.sidebar.slider("pH", 2.5, 4.5, 3.2)
sulphates = st.sidebar.slider("Sulphates (log)", 0.1, 2.0, 0.60)
alcohol = st.sidebar.slider("Alcohol (%)", 8.0, 15.0, 10.5)

# ----------------------------------
# Create Input DataFrame
# ----------------------------------
input_data = pd.DataFrame({
    "fixed acidity": [fixed_acidity],
    "volatile acidity": [volatile_acidity],
    "citric acid": [citric_acid],
    "residual sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free sulfur dioxide": [free_sulfur_dioxide],
    "total sulfur dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol]
})

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔮 Predict Wine Quality"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    quality = int(np.round(prediction[0]))

    st.markdown(
        f"<div class='result-card'>🍷 Wine Quality Score : {quality}</div>",
        unsafe_allow_html=True
    )

    if quality >= 7:
        st.markdown("🏆 **Excellent Quality Wine**")
        st.balloons()
    elif quality >= 5:
        st.markdown("👍 **Average Quality Wine**")
    else:
        st.markdown("⚠️ **Low Quality Wine**")

# ----------------------------------
# Footer
# ----------------------------------
st.divider()
st.markdown("<div class='footer'>Built with ❤️ using Streamlit & Machine Learning</div>", unsafe_allow_html=True)
