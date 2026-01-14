import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

model=pickle.load(open("model.pkl","rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))

def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]','',text)
    tokens=text.split()
    tokens=[word for word in tokens if word not in stop_words]
    return " ".join(tokens)

st.set_page_config(page_title="Resume Analyzer tool",layout="centered")

st.title("Resume Analyzer tool")
st.write("Paste your resume text below to predict the job category")

resume_text=st.text_area("Resume text", height=250)

if st.button("Analyze resume"):
    if resume_text.strip()=="":
        st.warning("please paste resume text")
    else:
        cleaned=clean_text(resume_text)
        vector=vectorizer.transform([cleaned])
        prediction=model.predict(vector)[0]
        st.success(f"predicted job category is **{prediction}**")