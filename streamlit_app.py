from pathlib import Path
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Spam or Ham", page_icon="📨", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "vectorizer.pkl"

@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Theme-aware CSS: all surfaces and text use Streamlit theme variables.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp {font-family:'DM Sans',sans-serif;background:radial-gradient(ellipse at 10% 0%,color-mix(in srgb,var(--primary-color) 10%,transparent),transparent 38%),var(--background-color);}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1120px;padding-top:2rem;padding-bottom:2.5rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.03em}
.hero{padding:1.6rem 1.8rem;border:1px solid color-mix(in srgb,var(--text-color) 12%,transparent);border-radius:22px;background:var(--secondary-background-color);margin-bottom:1.2rem}
.eyebrow{color:var(--primary-color);text-transform:uppercase;letter-spacing:.13em;font-size:.72rem;font-weight:800;margin-bottom:.5rem}
.hero-title{color:var(--text-color);font-family:'Space Grotesk',sans-serif;font-size:clamp(2rem,5vw,3.2rem);font-weight:700;line-height:1.1;margin:0}
.hero-copy,.helper{color:var(--text-color);opacity:.72;line-height:1.6}
.hero-copy{max-width:720px;margin:.7rem 0 0}
.tile{border:1px solid color-mix(in srgb,var(--text-color) 12%,transparent);border-radius:15px;background:var(--secondary-background-color);padding:1rem;height:100%}
.tile-label{font-size:.75rem;opacity:.65;color:var(--text-color);margin-bottom:.3rem}
.tile-title{font-weight:700;color:var(--text-color)}
.result{border-radius:15px;padding:1.1rem 1.25rem;margin:.8rem 0;border:1px solid color-mix(in srgb,var(--text-color) 14%,transparent);border-left:5px solid}
.spam{background:color-mix(in srgb,#ef4444 13%,var(--secondary-background-color));border-left-color:#ef4444}
.ham{background:color-mix(in srgb,#16a34a 13%,var(--secondary-background-color));border-left-color:#16a34a}
.result-title{font-size:1.25rem;font-weight:800;color:var(--text-color);margin:0 0 .25rem}
.result-copy{font-size:.9rem;color:var(--text-color);opacity:.75;margin:0}
div[data-testid="stTextArea"] textarea{border-radius:13px;min-height:190px;background:var(--secondary-background-color);color:var(--text-color);font-size:.96rem}
div[data-testid="stTextArea"] textarea:focus{border-color:var(--primary-color);box-shadow:0 0 0 1px var(--primary-color)}
div.stButton>button,div[data-testid="stFormSubmitButton"]>button{border-radius:12px;min-height:2.8rem;font-weight:700;transition:transform .18s ease}
div.stButton>button:hover,div[data-testid="stFormSubmitButton"]>button:hover{transform:translateY(-1px)}
a.profile-link{display:inline-block;padding:.7rem 1rem;border:1px solid color-mix(in srgb,var(--text-color) 16%,transparent);border-radius:12px;color:var(--text-color)!important;text-decoration:none!important;font-weight:700;background:var(--secondary-background-color);margin:.25rem .35rem .25rem 0}
a.profile-link:hover{border-color:var(--primary-color);color:var(--primary-color)!important}
.footer{text-align:center;color:var(--text-color);opacity:.55;font-size:.82rem;padding-top:1.5rem}
[data-testid="stMetric"]{background:var(--secondary-background-color);border:1px solid color-mix(in srgb,var(--text-color) 12%,transparent);padding:.8rem 1rem;border-radius:14px}
@media(max-width:700px){.block-container{padding:1rem}.hero{padding:1.2rem}}
</style>
""", unsafe_allow_html=True)

logo = ROOT / "assets" / "logo.png"
c1, c2 = st.columns([1, 5], vertical_alignment="center")
with c1:
    if logo.exists():
        st.image(str(logo), width=88)
    else:
        st.markdown("## 📨")
with c2:
    st.markdown("""
    <div class="eyebrow">MESSAGE INTELLIGENCE · NLP + MACHINE LEARNING</div>
    <div class="hero-title">Spam or Ham<span style="color:var(--primary-color)">.</span></div>
    <p class="hero-copy">A smarter way to check whether an email or message looks like spam.
    Paste your text, run the classifier, and review the prediction.</p>
    """, unsafe_allow_html=True)

st.write("")
checker, about = st.tabs(["✦  Spam checker", "ⓘ  About"])

with checker:
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.subheader("Analyze a message")
        st.caption("Paste the email or message you want to inspect.")
        with st.form("spam_form", clear_on_submit=False):
            message = st.text_area("Message content", placeholder="Example: Congratulations! You have won a free prize. Click here to claim it...", label_visibility="collapsed", height=210)
            submitted = st.form_submit_button("🔎  Analyze message", type="primary", use_container_width=True)
        st.markdown('<p class="helper">Tip: Include the full message for a more useful prediction. Never share passwords, OTPs, or sensitive information.</p>', unsafe_allow_html=True)

    with right:
        st.subheader("How it works")
        for label, title, description in [
            ("01 · TEXT PROCESSING", "TF-IDF Vectorization", "Converts message text into numerical features."),
            ("02 · CLASSIFICATION", "Multinomial Naive Bayes", "Uses learned word patterns to classify the message."),
            ("03 · PREDICTION", "Label + confidence", "Displays the predicted class and probability estimate.")
        ]:
            st.markdown(f'<div class="tile"><div class="tile-label">{label}</div><div class="tile-title">{title}</div><p class="helper">{description}</p></div><br>', unsafe_allow_html=True)

    if submitted:
        if not message.strip():
            st.warning("Please enter a message before running the analysis.")
        else:
            try:
                model, vectorizer = load_artifacts()
                features = vectorizer.transform([message.strip()])
                prediction = model.predict(features)[0]
                confidence = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    classes = list(getattr(model, "classes_", []))
                    confidence = float(probabilities[classes.index(prediction)]) if classes and prediction in classes else float(np.max(probabilities))

                is_spam = str(prediction).strip().lower() in {"1", "spam", "true"}
                kind = "spam" if is_spam else "ham"
                heading = "🚨 Spam detected" if is_spam else "✅ Looks like not spam"
                copy = ("This message was classified as spam. Be cautious with links, attachments, and requests for personal information."
                        if is_spam else "This message was classified as not spam. Still verify unexpected links and requests before taking action.")
                st.divider()
                st.subheader("Analysis result")
                st.markdown(f'<div class="result {kind}"><p class="result-title">{heading}</p><p class="result-copy">{copy}</p></div>', unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Predicted category", "Spam" if is_spam else "Not Spam")
                with m2:
                    st.metric("Model confidence", f"{confidence*100:.1f}%" if confidence is not None else "Unavailable")
                if confidence is not None:
                    st.progress(min(max(confidence, 0.0), 1.0), text="Confidence for predicted class")
                st.caption("Confidence is an estimate, not a guarantee. This model can make mistakes.")
            except FileNotFoundError:
                st.error("Model files are missing. Check that models/spam_model.pkl and models/vectorizer.pkl are present.")
            except Exception as exc:
                st.error(f"Unable to analyze this message: {exc}")

with about:
    st.subheader("About Spam or Ham")
    st.markdown("""
    **Spam or Ham** is a lightweight email and message classification application built with
    Python, Streamlit, and Natural Language Processing.

    The app converts text into TF-IDF features and uses a trained **Multinomial Naive Bayes**
    classifier to predict whether a message is spam or not spam. It demonstrates a practical
    machine-learning workflow through a simple, accessible interface.
    """)
    st.markdown("#### Technology stack")
    cols = st.columns(4)
    for col, title, desc in zip(cols, ["Python", "Streamlit", "Scikit-learn", "Naive Bayes"], ["Core language", "Web interface", "ML + TF-IDF", "Text classification"]):
        with col:
            st.markdown(f'<div class="tile"><div class="tile-title">{title}</div><p class="helper">{desc}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("#### About the developer")
    st.markdown("""
    Hi, I'm **Diwakar Jha**, an MCA student specializing in Artificial Intelligence and
    Machine Learning. I enjoy building practical applications involving machine learning,
    NLP, and generative AI.
    """)
    st.markdown("""
    <a class="profile-link" href="https://www.linkedin.com/in/diwakar-jha-064130229/" target="_blank" rel="noopener noreferrer">↗ LinkedIn · Diwakar Jha</a>
    <a class="profile-link" href="https://github.com/Stfu-diwakar" target="_blank" rel="noopener noreferrer">↗ GitHub · Stfu-diwakar</a>
    """, unsafe_allow_html=True)
    st.caption("Educational ML demonstration only. Do not rely on it as the sole security filter for important communications.")

st.markdown('<div class="footer">Built with Python, Streamlit &amp; Machine Learning · © Diwakar Jha</div>', unsafe_allow_html=True)
