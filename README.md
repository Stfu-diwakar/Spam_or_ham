# Spam or Ham — Machine Learning Spam Classifier

A clean, interactive machine-learning application that classifies email and message text as **Spam** or **Not Spam** using TF-IDF and a trained Multinomial Naive Bayes model.

<p align="center">
  <a href="https://spamorham-d.streamlit.app/"><strong>🚀 Try the Live App</strong></a>
  &nbsp; · &nbsp;
  <a href="https://github.com/Stfu-diwakar/Spam_or_ham"><strong>💻 View Source</strong></a>
</p>

## Overview

Spam or Ham demonstrates a practical NLP classification workflow through a simple Streamlit interface. Users can paste a message, run a prediction, and view the predicted category with the model's confidence estimate.

## Features

- Spam / not-spam message classification
- Confidence display when probability estimates are available
- Theme-aware interface that adapts to Streamlit light and dark themes
- Responsive layout for desktop and smaller screens
- Dedicated About section with project details and developer profile links
- Pre-trained model and TF-IDF vectorizer loaded from the models directory

## How it works

1. **Input:** The user enters an email or message.
2. **Vectorization:** The saved TF-IDF vectorizer converts text into numerical features.
3. **Classification:** The trained Multinomial Naive Bayes model predicts the class.
4. **Result:** The app displays the predicted category and confidence, if supported by the model.

## Tech stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Interactive web interface |
| Scikit-learn | Machine-learning model and TF-IDF |
| NumPy | Numerical operations |
| Pillow | Image support for project assets |

## Project structure

~~~text
Spam_or_ham/
├── assets/
│   └── logo.png
├── models/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
├── streamlit_app.py
├── requirements.txt
└── README.md
~~~

## Run locally

### 1. Clone the repository

~~~bash
git clone https://github.com/Stfu-diwakar/Spam_or_ham.git
cd Spam_or_ham
~~~

### 2. Create a virtual environment (recommended)

~~~bash
python -m venv .venv
~~~

Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

macOS / Linux:

~~~bash
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 4. Start the app

~~~bash
streamlit run streamlit_app.py
~~~

## Developer

**Diwakar Jha** · MCA (Artificial Intelligence & Machine Learning)

- [LinkedIn](https://www.linkedin.com/in/diwakar-jha-064130229/)
- [GitHub](https://github.com/Stfu-diwakar)

## Important note

This is an educational machine-learning demonstration. Predictions and confidence values are estimates and can be incorrect. Do not use the app as the sole security control for filtering important or sensitive communications.

---

If you find this project useful, feel free to ⭐ the repository.
