import streamlit as st
import pickle 
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

vectorizer=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('SMS Spam classifier')
inp_text=st.text_area('enter the message')
ps=PorterStemmer()
def text_transform(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z0-9\s]','',text)
    tokens=nltk.word_tokenize(text)
    stop_words=stopwords.words('english')
    cleaned_tokens=[word for word in tokens if word not in stop_words ]
    y=[]
    for word in cleaned_tokens:
        y.append(ps.stem(word))
    return ' '.join(y)




def predict_msg(text):
    transformed_text1=text_transform(text)

    transformed_text=vectorizer.transform([transformed_text1])

    prediction=model.predict(transformed_text)[0] 
    return "spam" if prediction==1 else "ham" 

if st.button('predict'):
    if inp_text.strip=="":
        st.warning("enter a messagge")
    else:
        result=predict_msg(inp_text)
        if result=="spam":
            st.error("It is a spam message")
        else:
            st.success("It is not a spam message")
