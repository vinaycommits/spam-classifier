import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('spam.csv',encoding='ISO-8859-1')
df.head()

# Data cleaning
df.columns=df.columns.str.lower()
df.columns=df.columns.str.strip()
df.columns=df.columns.str.replace(' ','')
#drop last 3 columns
df.drop(columns=['unnamed:2','unnamed:3','unnamed:4'],inplace=True)
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
df.duplicated().sum()
df=df.drop_duplicates(keep='first')
df.shape

#EDA
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct='%.2f')
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
df['num_characters']=df['text'].apply(len)
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
sns.pairplot(df,hue='target')
plt.show()
df1=df[['target','num_characters','num_words','num_sentences']]
sns.heatmap(df1.corr(),cmap='coolwarm',annot=True)
plt.show()
#Text Preprocessing
# lower case
#  tokenisation
#  removing special characters
#  removing stop words and punctuations
#  stemming
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def text_transform(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z0-9\s]','',text)
    tokens=nltk.word_tokenize(text)
    stop_words=stopwords.words('english')
    cleaned_tokens=[word for word in tokens if word not in stop_words]
    y=[]
    for word in cleaned_tokens:
        y.append(ps.stem(word))
    return ' '.join(y)

df['transformed_text']=df['text'].apply(text_transform)
from wordcloud import WordCloud
spam_wordcloud=WordCloud(width=800,height=800,background_color='white').generate(df[df['target']==1]['transformed_text'].str.cat(sep=' '))
plt.figure(figsize=(6,6))
plt.imshow(spam_wordcloud)

ham_wordcloud=WordCloud(width=800,height=800,background_color='white').generate(df[df['target']==0]['transformed_text'].str.cat(sep=' '))
plt.figure(figsize=(6,6))
plt.imshow(ham_wordcloud)

spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

from collections import Counter
df_spam=pd.DataFrame(Counter(spam_corpus).most_common(30),columns=['word','count'])
print(df_spam)

sns.barplot(x='word',y='count',data=df_spam)
plt.xticks(rotation=90)
plt.show()

ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

df_ham=pd.DataFrame(Counter(ham_corpus).most_common(30),columns=['word','count'])

sns.barplot(x='word',y='count',data=df_ham)
plt.xticks(rotation=90)
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

counter=CountVectorizer(max_features=5000)

x = counter.fit_transform(df['transformed_text'])  
y=df['target']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)
print(accuracy_score(y_pred_lr,y_test))
print(confusion_matrix(y_pred_lr,y_test))
print(precision_score(y_pred_lr,y_test))

from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(x_train,y_train)
y_pred_mnb=mnb.predict(x_test)
print(accuracy_score(y_pred_mnb,y_test))
print(confusion_matrix(y_pred_mnb,y_test))
print(precision_score(y_pred_mnb,y_test))



import pickle
pickle.dump(counter,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))

