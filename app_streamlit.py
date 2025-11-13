import streamlit as st
import requests

API_URL = "http://localhost:5000"

st.title("Skincare Review Summarizer (RAG + GPT)")

query = st.text_input("Enter your skincare concern:")

if st.button("Summarize"):
    if query.strip():
        response = requests.get(API_URL, params={"question": query})
        data = response.json()
        st.write("### ✅ Summary")
        st.write(data["summary"])
    else:
        st.error("Please enter a question.")
