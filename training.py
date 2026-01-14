import pandas as pd
import re
import nltk
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# nltk.download('stopwords')
# stop_words=set(stopwords.words('english'))

stop_words=ENGLISH_STOP_WORDS

def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]','',text)
    tokens=text.split()
    tokens=[word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df=pd.read_csv("resume.csv")
df["cleaned_resume"]=df["Resume"].apply(clean_text)

x=df['cleaned_resume']
y=df['Category']

vectorizer=TfidfVectorizer(max_features=3000)
x_vec=vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x_vec, y, test_size=0.2)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

nb_model=MultinomialNB()
nb_model.fit(x_train,y_train)
nb_pred=nb_model.predict(x_test)

acc=accuracy_score(y_test, y_pred)
nb_acc=accuracy_score(y_test, nb_pred)

print("accuracy in logistic regression: ", acc)
print("classification report in logistic regression: \n")
print(classification_report(y_test, y_pred))

print("accuracy in naive bayes: ", nb_acc)
print("classification report in naive bayes: \n")
print(classification_report(y_test, nb_pred))


cm=confusion_matrix(y_test,y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',cmap="Blues")
plt.title("confusion matrix for logistic regression")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

best_model=model if acc>=nb_acc else nb_model

# pickle.dump(model, open("model.pkl","wb"))
# pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

with open("model.pkl","wb") as f:
    pickle.dump(best_model,f)
with open("vectorizer.pkl","wb") as f:
    pickle.dump(vectorizer,f)


print("model and vectorizer saved")