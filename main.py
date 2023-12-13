# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 00:18:58 2023

@author: CS Wizard
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 23:26:12 2023

@author: CS Wizard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

df.sample(10)

df.info()

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace = True)

print(df[df.isnull()])
print(df.isnull().sum())

df.rename(columns={'v1':'target','v2':'text'},inplace = True)
df.sample(5)

#checking for duplicate...
df.duplicated().sum()

df = df.drop_duplicates(keep='first')


df['target'].value_counts()

#Data is imbalanced so we have to take care of this 

from matplotlib import pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


plt.hist(df['target'].value_counts(),bins=24,color='skyblue',edgecolor='black')
plt.xlabel('ham')
plt.ylabel('spam')
plt.show()

data=["ham","spam",]
plt.bar(data,df['target'].value_counts(),color=('red','blue'))
plt.legend(['ham','spam','other'],loc='upper right')
plt.xlabel('email')
plt.ylabel('counts')
plt.show()

import nltk
nltk.download('punkt')
nltk.download('stopwords')

#Creating new Feature (num_of_Character)
df['num_of_character'] = df['text'].apply(len)

#for num of words
df['num_word'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

df['num_sentence'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

df[['num_of_character','num_word','num_sentence']].describe()

#Analysis where target == 0 i.e. ham......
df[df['target'] == 0][['num_of_character','num_word','num_sentence']].describe()

#Analysis where target == 1 i.e. spam...
df[df['target'] == 1][['num_of_character','num_word','num_sentence']].describe()


#Now see these conclusions by graphically
import seaborn as sns 
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_of_character'])
sns.histplot(df[df['target'] == 1]['num_of_character'])
plt.grid()


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_word'])
sns.histplot(df[df['target'] == 1]['num_word'],color='red')
plt.grid()


sns.pairplot(df,hue='target')
plt.grid()


sns.heatmap(df[['num_of_character','num_word','num_sentence']].corr(),annot=True)


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

df['text'][10]

transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")

df['transformed_text'] = df['text'].apply(transform_text)
df.head(5)


from wordcloud import WordCloud

wc = WordCloud(width=500,height=500,min_font_size=10,background_color = 'white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize = (15,6))
plt.imshow(spam_wc)


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)

########Model Building

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features = 4000)
x = tfidf.fit_transform(df['transformed_text']).toarray()
x.shape

y = df['target'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state = 20)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

gnb = GaussianNB()
mnb = MultinomialNB(alpha=1.0)
bnb = BernoulliNB()
gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

mnb.fit(x_train,y_train)
y_pred2 = mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

bnb.fit(x_train,y_train)
y_pred3 = bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

import pickle
pickle.dump(tfidf,open('nlp_vectorizer1.pkl','wb'))
pickle.dump(mnb,open('nlp_model1.pkl','wb'))

##Testing
pred = mnb.predict(x_test)
pred

y_test

