
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="HomeValue | House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("house_price_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Brown theme
st.markdown("""
<style>
.stApp {
    background-color: #2B1B14;
    color: #F5E6D3;
}
h1, h2, h3, p, label {
    color: #F5E6D3 !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #634735;
    color: white;
}
.stButton > button {
    background-color: #A87955;
    color: white;
    border-radius: 10px;
    border: none;
    width: 100%;
    height: 50px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #C49A78;
    color: #2B1B14;
}
</style>
""", unsafe_allow_html=True)

st.title("🏡 HomeValue")
st.caption("HOUSE PRICE PREDICTION")
st.markdown("### Find your home's estimated value.")
st.write("Enter your property details below.")

# Get locations from trained model
locations = model.named_steps[
    "columntransformer"
].named_transformers_["onehot"].categories_[0]

location = st.selectbox("Select Location", locations)

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input(
        "Total Square Feet",
        min_value=100.0,
        value=1000.0
    )

with col2:
    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=20,
        value=2
    )

bath = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=20,
    value=2
)

if st.button("PREDICT HOUSE PRICE"):
    if sqft <= 0:
        st.error("Please enter a valid area.")
    else:
        input_data = pd.DataFrame([{
            "location": location,
            "total_sqft": sqft,
            "bath": bath,
            "bhk": bhk
        }])

        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")
        st.markdown("### Estimated House Price")
        st.markdown(
            f"<h1 style='color:#D6AD8B'>₹ {prediction:.2f} Lakhs</h1>",
            unsafe_allow_html=True
        )

