import streamlit as st
from src.pipelines.predict import predict_sentiment
from src.features.text_cleaning import DataTransformation

st.title("🎬 IMDb Sentiment Analysis")
st.write("Enter a movie review and get the sentiment prediction.")


# User input
user_input = st.text_area("Movie Review", placeholder="Type your review here...")


if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a review first.")
    else:
        # Preprocess + predict
        cleaned_text = DataTransformation().clean_text(user_input)

        label = predict_sentiment([cleaned_text])
        st.success(f"Prediction: {label}")
