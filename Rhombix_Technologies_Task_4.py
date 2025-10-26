import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# ------------------------------------------
# 🧹 Text Cleaning Function
# ------------------------------------------
def clean_text(text):
    text = str(text).lower()                       # lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>', '', text)              # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)           # keep only letters
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ------------------------------------------
# 🏠 Streamlit Page Config
# ------------------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detection System")
st.markdown("""
This web app detects whether a given **news article** is *Fake* or *Real* using Natural Language Processing (NLP)  
and a trained **Multinomial Naive Bayes** model.
""")

st.markdown("---")

# ------------------------------------------
# 📂 Upload Dataset
# ------------------------------------------
st.header("📁 Step 1: Upload Datasets")

col1, col2 = st.columns(2)
with col1:
    fake_file = st.file_uploader("Upload Fake News Dataset (CSV)", type=["csv"])
with col2:
    true_file = st.file_uploader("Upload True News Dataset (CSV)", type=["csv"])

# ------------------------------------------
# 🧩 Data Preparation
# ------------------------------------------
if fake_file and true_file:
    fake = pd.read_csv(fake_file)
    true = pd.read_csv(true_file)

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true], axis=0)
    data = data.sample(frac=1).reset_index(drop=True)

    st.success(f"✅ Dataset Loaded Successfully — {len(data)} total articles")

    st.subheader("🔎 Data Preview")
    st.dataframe(data.head(10))

    # Clean text
    st.info("🧹 Cleaning text data... please wait.")
    data["text"] = data["text"].astype(str).apply(clean_text)

    st.success("✅ Text cleaning complete!")

    # ------------------------------------------
    # 📊 Visualizations
    # ------------------------------------------
    st.subheader("📊 Data Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="label", data=data, palette="Set2", ax=ax)
    ax.set_xticklabels(["Fake News (0)", "True News (1)"])
    ax.set_title("Fake vs Real News Count")
    st.pyplot(fig)

    # ------------------------------------------
    # 🧠 Model Training
    # ------------------------------------------
    st.header("🧠 Step 2: Model Training")

    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.01)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.25, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric(label="🎯 Model Accuracy", value=f"{acc * 100:.2f}%")

    # Classification Report
    st.subheader("📈 Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    st.markdown("---")

    # ------------------------------------------
    # 🧾 Prediction Section
    # ------------------------------------------
    st.header("🔍 Step 3: Test Your Own News")

    user_text = st.text_area("✍️ Enter a news article or headline:")

    if st.button("Analyze News"):
        if user_text.strip():
            cleaned = clean_text(user_text)
            vec_input = vectorizer.transform([cleaned])
            prediction = model.predict(vec_input)[0]

            if prediction == 0:
                st.error("🚨 This looks like **FAKE NEWS!** ❌")
            else:
                st.success("✅ This appears to be **REAL NEWS!** 👍")
        else:
            st.warning("⚠️ Please enter some text to analyze.")
else:
    st.warning("📌 Please upload both Fake.csv and True.csv files to continue.")

st.markdown("---")
st.caption("Developed by Rhombix Technologies • Powered by Streamlit + Scikit-learn + NLTK")
