# 📧 !-SPAM — Spam Email Classifier

A modern Machine Learning web app that detects whether a message is **Spam 🚫** or **Not Spam 📩** using Natural Language Processing (NLP).

---

## 🌐 Live Demo

🔗 **Website:** 

---

## 🚀 Features

* 📩 Detect spam emails/messages instantly
* 🧠 Built using Machine Learning (Naive Bayes + TF-IDF)
* 📊 Shows prediction confidence score
* 🎨 Clean and modern UI (Streamlit + custom styling)
* 🖼️ Custom logo and branding
* ⚡ Fast and lightweight

---

## 🧠 How It Works

1. Text input is converted into numerical features using **TF-IDF Vectorization**
2. A trained **Naive Bayes model** predicts whether the message is spam
3. The app displays:

   * Prediction (Spam / Not Spam)
   * Confidence score

---

## 🏗️ Project Structure

```
spam-email-classifier/
│
├── app.py                 # Streamlit UI
├── train.py               # Model training script
│
├── models/                # Saved model files
│   ├── spam_model.pkl
│   ├── vectorizer.pkl
│
├── data/                  # Training datasets (multiple CSVs)
│
├── assets/
│   └── logo.png           # App logo
│
├── requirements.txt       # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train the model

```bash
python train.py
```

This will generate:

* `spam_model.pkl`
* `vectorizer.pkl`

---

### 4️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📊 Model Details

* Algorithm: **Multinomial Naive Bayes**
* Text Processing: **TF-IDF Vectorization**
* Handles multiple datasets with different formats
* Includes:

  * Data cleaning
  * Label normalization
  * Dataset balancing

---

## 🎯 Example

**Input:**

```
Congratulations! You’ve won a free iPhone. Click here now!
```

**Output:**

```
🚫 Spam Detected (98.45%)
```

---

## 🔥 Future Improvements

* 🤖 Explain why a message is spam
* 📊 Visual analytics dashboard
* 🌍 Multi-language support
* 🧠 Upgrade to advanced models (SVM / Deep Learning)
* 📁 Upload email files

---

## 🧑‍💻 Author

Made with ❤️ by DWKR

---

## ⭐ Contribute

Feel free to fork this repo and improve it!

---

## 📜 License

This project is open-source and available under the MIT License.
