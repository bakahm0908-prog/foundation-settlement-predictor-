
import streamlit as st

st.set_page_config(
    page_title="Foundation Settlement Predictor",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Foundation Settlement Predictor")

st.subheader(
    "AI-Based Prediction of Shallow Foundation Settlement"
)

st.markdown("""
Welcome to our academic geotechnical engineering platform.

This platform aims to predict foundation settlement using
machine learning and verified experimental data.
""")

st.divider()

st.header("Foundation and Soil Parameters")

col1, col2 = st.columns(2)

with col1:
    B = st.number_input(
        "Foundation Width B (m)",
        min_value=0.01,
        value=2.0
    )

    q = st.number_input(
        "Applied Pressure q (kPa)",
        min_value=0.01,
        value=150.0
    )

    N = st.number_input(
        "Average SPT N Value",
        min_value=0.0,
        value=20.0
    )

with col2:
    LB = st.number_input(
        "Length / Width Ratio (L/B)",
        min_value=0.01,
        value=1.0
    )

    DfB = st.number_input(
        "Embedment Ratio (Df/B)",
        min_value=0.0,
        value=0.5
    )

st.divider()

if st.button("Predict Settlement", use_container_width=True):
    st.warning(
        "The prediction model is currently under development. "
        "No prediction will be generated until the trained model "
        "has been scientifically evaluated."
    )

st.divider()

st.subheader("Research Background")

st.write("""
The initial research model is based on the work of
Mohamed A. Shahin (2003), concerning settlement prediction
of shallow foundations on cohesionless soils using
artificial neural networks.
""")

st.caption(
    "Academic research prototype | Model development in progress"
)
