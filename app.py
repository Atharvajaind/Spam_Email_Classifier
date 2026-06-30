import streamlit as st
import joblib

# Load Model
model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Page Settings
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# Sidebar
st.sidebar.title("Project Information")
st.sidebar.write("**Developer:** Atharva Jaind")
st.sidebar.write("**Course:** MCA - AI & Data Science")
st.sidebar.write("**Algorithm:** Multinomial Naive Bayes")
st.sidebar.write("**Vectorizer:** TF-IDF")
st.sidebar.write("**Model Accuracy:** 96%")

# Main Title
st.title("📧 AI Spam Email Classifier")

st.write(
    "This application predicts whether an Email or SMS is **Spam** or **Not Spam** using Machine Learning."
)

# User Input
message = st.text_area(
    "Enter your Email or SMS Message",
    height=180
)

# Predict Button
if st.button("🔍 Analyze Message"):

    if message.strip() == "":
        st.warning("⚠ Please enter a message.")

    else:
        transformed = vectorizer.transform([message])

        prediction = model.predict(transformed)

        if prediction[0] == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")