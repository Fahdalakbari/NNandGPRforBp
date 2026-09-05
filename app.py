import streamlit as st
from utils import load_models, predict

st.set_page_config(
    page_title="Bubble Point Pressure Prediction",
    page_icon="🛢️",
    layout="centered"
)

# ==========================================================
# CSS (PUT IT HERE)
# ==========================================================

st.markdown("""
<style>

/* ==========================================================
   Background
========================================================== */

.stApp{
    background: linear-gradient(135deg,#F5F9FF,#EAF3FF);
}

/* ==========================================================
   Main container
========================================================== */

.block-container{
    max-width:950px;
    padding-top:1.2rem;
    padding-bottom:2rem;
}

/* ==========================================================
   Main title
========================================================== */

h1{
    color:#004C99 !important;
    font-size:44px !important;
    font-weight:800 !important;
    text-align:center;
}

/* ==========================================================
   Subtitle
========================================================== */

p{
    color:#1E5AA8 !important;
    font-size:22px !important;
    font-weight:500 !important;
}

/* ==========================================================
   Input Labels
========================================================== */

div[data-testid="stTextInput"] label {

    color:#004C99 !important;

    font-size:24px !important;

    font-weight:800 !important;

}

/* ==========================================================
   Text inside input boxes
========================================================== */

div[data-testid="stTextInput"] input {

    font-size:24px !important;

    font-weight:800 !important;

    color:#003366 !important;

    border-radius:12px !important;

    border:2px solid #BFD7F2 !important;

}

/* ==========================================================
   Buttons
========================================================== */

button[kind="secondary"]{

    background:#0066CC !important;

    color:white !important;

    font-size:22px !important;

    font-weight:700 !important;

    border-radius:12px !important;

    height:58px !important;

    border:none !important;

}

button[kind="secondary"] p{

    color:white !important;

    font-size:22px !important;

    font-weight:700 !important;

}

button[kind="secondary"]:hover{

    background:#004C99 !important;

}

button[kind="secondary"]:hover p{

    color:white !important;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def initialize():
    return load_models()

nn_model, gpr_model, scalers = initialize()

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.image("header.PNG", width=700)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# Default Values
# ==========================================================

DEFAULTS = {

    "T": "132",

    "N2": "0.0",
    "C1": "34.53",
    "C2": "7.24",
    "C3": "5.31",
    "C4": "2.79",
    "C5": "1.91",
    "C6": "3.48",
    "C7": "43.44",
    "MWC7": "200",
    "H2S": "0.0",
    "CO2": "1.3",
}

EMPTY = {k: "" for k in DEFAULTS}

# Initialize Session State

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ==========================================================
# Header
# ==========================================================

st.markdown(
"""
<h1>Bubble Point Pressure Prediction</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
Predict **Bubble Point Pressure (Pb)** from reservoir-fluid composition using
either a **Neural Network (NN)** or **Gaussian Process Regression (GPR)**.
"""
)

st.divider()

# ==========================================================
# Model selection
# ==========================================================

selected_model = st.radio(
    "Select prediction model",
    ["Neural Network (NN)", "Gaussian Process Regression (GPR)"],
    horizontal=True
)

st.info(f"Selected model: {selected_model}")

st.divider()

# ==========================================================
# Inputs
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    st.session_state["T"] = st.text_input(
        "Temperature (°F)",
        value=st.session_state["T"]
    )

    st.session_state["N2"] = st.text_input(
        "N₂ (mol%)",
        value=st.session_state["N2"]
    )

    st.session_state["C1"] = st.text_input(
        "C1 (mol%)",
        value=st.session_state["C1"]
    )

    st.session_state["C2"] = st.text_input(
        "C2 (mol%)",
        value=st.session_state["C2"]
    )

    st.session_state["C3"] = st.text_input(
        "C3 (mol%)",
        value=st.session_state["C3"]
    )

    st.session_state["C4"] = st.text_input(
        "C4 (mol%)",
        value=st.session_state["C4"]
    )


with col2:

    st.session_state["C5"] = st.text_input(
        "C5 (mol%)",
        value=st.session_state["C5"]
    )

    st.session_state["C6"] = st.text_input(
        "C6 (mol%)",
        value=st.session_state["C6"]
    )

    st.session_state["C7"] = st.text_input(
        "C7+ (mol%)",
        value=st.session_state["C7"]
    )

    st.session_state["MWC7"] = st.text_input(
        "MW C7+",
        value=st.session_state["MWC7"]
    )

    st.session_state["H2S"] = st.text_input(
        "H₂S (mol%)",
        value=st.session_state["H2S"]
    )

    st.session_state["CO2"] = st.text_input(
        "CO₂ (mol%)",
        value=st.session_state["CO2"]
    )

st.divider()

# ==========================================================
# Buttons
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    predict_button = st.button(
        "Predict",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "Clear",
        use_container_width=True
    )

# ==========================================================
# Clear
# ==========================================================

if clear_button:

    for key in EMPTY:
        st.session_state[key] = ""

    st.rerun()

# ==========================================================
# Prediction
# ==========================================================

if predict_button:

    try:

        T = float(st.session_state["T"])
        N2 = float(st.session_state["N2"])
        C1 = float(st.session_state["C1"])
        C2 = float(st.session_state["C2"])
        C3 = float(st.session_state["C3"])
        C4 = float(st.session_state["C4"])
        C5 = float(st.session_state["C5"])
        C6 = float(st.session_state["C6"])
        C7 = float(st.session_state["C7"])
        MWC7 = float(st.session_state["MWC7"])
        H2S = float(st.session_state["H2S"])
        CO2 = float(st.session_state["CO2"])

    except ValueError:

        st.error("Please enter numeric values in all fields.")
        st.stop()

    with st.spinner("Predicting Bubble Point Pressure..."):

        result = predict(
            selected_model,
            nn_model,
            gpr_model,
            scalers,
            T,
            N2,
            C1,
            C2,
            C3,
            C4,
            C5,
            C6,
            C7,
            MWC7,
            H2S,
            CO2,
        )

    st.divider()

    st.success(
        f"""
### Predicted Bubble Point Pressure

# **{result:.2f} psi**

Model: **{selected_model}**
"""
    )
