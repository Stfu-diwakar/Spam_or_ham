import streamlit as st
import pickle
from PIL import Image

# -------------------------------
# LOAD MODEL
# -------------------------------
model = pickle.load(open("models/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="!-SPAM",
    page_icon="📧",
    layout="centered"
)

# -------------------------------
# GOOGLE FONT
# -------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    font-family: 'Poppins', sans-serif;
}

/* Card */
.card {
    padding: 25px;
    border-radius: 18px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}

/* Text area */
textarea {
    background-color: #020617 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, #f43f5e, #e11d48);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton button:hover {
    transform: scale(1.05);
    transition: 0.2s;
}

/* Result boxes */
.spam {
    background: #7f1d1d;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
}

.ham {
    background: #14532d;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: #64748b;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER WITH LOGO
# -------------------------------
logo = Image.open("assets/logo.png")

col1, col2 = st.columns([1, 4])

with col1:
    st.image(logo, width=80)

with col2:
    st.markdown("""
    <h1 style='
        font-size: 42px;
        font-weight: 800;
        color: #f43f5e;
        margin-bottom: 0;
    '>
        !-SPAM
    </h1>
    """, unsafe_allow_html=True)

st.markdown("""
<p style='color: #94a3b8; font-size:16px; margin-top: -10px;'>
Detect spam messages using Machine Learning
</p>
""", unsafe_allow_html=True)

# -------------------------------
# CARD START
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

user_input = st.text_area("✉️ Enter Email or Message")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Analyze Message"):
    if user_input.strip() != "":
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]

        confidence = max(prob)

        if prediction == 1:
            st.markdown(
                f'<div class="spam">🚫 Spam Detected ({confidence*100:.2f}%)</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="ham">📩 Not Spam ({confidence*100:.2f}%)</div>',
                unsafe_allow_html=True
            )

        st.progress(int(confidence * 100))

    else:
        st.warning("⚠️ Please enter some text.")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<hr>
<div style='text-align: center;'>

<p style="font-size:18px;">Made with ❤️ by <b>DWKR</b></p>

<a href="https://www.linkedin.com/in/diwakar-jha-064130229/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="40" style="margin:10px;">
</a>

<a href="https://github.com/Stfu-diwakar" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="40" style="margin:10px;">
</a>

</div>
""", unsafe_allow_html=True)