import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re, string

# -------------------------
# Utility functions
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

@st.cache_resource
def load_model():
    with open('models/model.pkl', 'rb') as f:
        saved = pickle.load(f)
    return saved['model'], saved['vectorizer']

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Quora Duplicate Question Detector", page_icon="🧠", layout="centered")

st.title("🧠 Quora Duplicate Question Detector")
st.write("Use this app to check if two questions mean the same thing!")

model, vectorizer = load_model()

tab1, tab2 = st.tabs(["📝 Single Pair", "📂 CSV Upload"])

# -------------------------
# Tab 1: Single Input
# -------------------------
with tab1:
    q1 = st.text_area("Enter first question", "")
    q2 = st.text_area("Enter second question", "")
    if st.button("Check"):
        if q1 and q2:
            q1_clean = clean_text(q1)
            q2_clean = clean_text(q2)
            q1_vec = vectorizer.transform([q1_clean])
            q2_vec = vectorizer.transform([q2_clean])
            sim = cosine_similarity(q1_vec, q2_vec)[0][0]
            pred = model.predict([[sim]])[0]
            result = "✅ Duplicate" if pred == 1 else "❌ Not Duplicate"
            st.success(f"**Result:** {result}")
            st.info(f"Cosine Similarity: {sim:.3f}")
        else:
            st.warning("Please enter both questions.")

# -------------------------
# Tab 2: CSV Upload
# -------------------------
with tab2:
    st.write("Upload a CSV file with **question1** and **question2** columns.")
    file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if file is not None:
        df = pd.read_csv(file)
        if 'question1' in df.columns and 'question2' in df.columns:
            df['question1'] = df['question1'].apply(clean_text)
            df['question2'] = df['question2'].apply(clean_text)

            st.write("✅ File successfully loaded!")
            if st.button("Run Predictions on Uploaded Data"):
                q1_vecs = vectorizer.transform(df['question1'])
                q2_vecs = vectorizer.transform(df['question2'])
                sims = [cosine_similarity(q1_vecs[i], q2_vecs[i])[0][0] for i in range(len(df))]
                preds = model.predict(pd.DataFrame(sims, columns=['similarity']))

                df['similarity'] = sims
                df['predicted_label'] = ['Duplicate' if p == 1 else 'Not Duplicate' for p in preds]

                st.dataframe(df.head(20))
                csv_out = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results CSV", csv_out, "predictions.csv", "text/csv")
        else:
            st.error("❌ CSV must have 'question1' and 'question2' columns.")
