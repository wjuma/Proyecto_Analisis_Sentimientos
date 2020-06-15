import pandas as pd
import random
import csv
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk # for text manipulation
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option("display.max_colwidth", 200)

columnas1 = ['text','user','target']
train = pd.read_csv('./documentos/lista-posi-nega-nueva2.csv', sep=';', encoding = "utf8", names=columnas1)
#train = train.drop(train[train['user']=="NE"].index) # eliman los NP

train =train.replace(2, 0)
print(train)
# gg=random.choice(train)
# print(gg)


pruebT = train[train['target'] == 1].head(10) # ayuda aver la cabezara de los 10 primero que calificamos como postivos


length_train = train['text'].str.len()
length_test = test['text'].str.len()
# la distrubucion de la logitud de los tweets es mas o menos la misma para ambos casos para los datos de train y test 
plt.hist(length_train, bins=20, label="train_tweets")
plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
plt.show()

#limpieza

combi = train.append(test, ignore_index=True)
#combi.shape

#(1542, 3)
# 


def remove_pattern(input_txt, pattern): # remueve patrones no deseados en el tweets
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

'''

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['text'], "@[\w]*") # remover el nombre del usuario 
combi.head()

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") # puntuacion numero y caracteres 
combi.head(10)
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    
combi['tidy_tweet'] = tokenized_tweet

#generar una visualizacion desde twiis
# palabras comunes 


all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
#pip install wordcloud
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


normal_words =' '.join([text for text in combi['tidy_tweet'][combi['target'] == 1]]) #nueve de palabras positivos 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['target'] == 0]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()



def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# Extraccion de hashtags de tweets positivos

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['target'] == 1])

# Extraccion de hashtags de tweets negativos
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['target'] == 0])

# unnesting list
HT_regular = sum(HT_regular,[]) #com ams impacto dataset
HT_negative = sum(HT_negative,[])


a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# Selccionando Top 20 hashtags mas frecuentes    
d = d.nlargest(columns="Count", n = 20)  #### hasthtag tweets postivos .....
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())}) #negativos 

# selecting top 20 most frequent hashtags
e = e.nlargest(columns="Count", n = 20)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

#extracion de caracteristicas desde los tweets limpios

# Bolsad de Palabras BOW
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
bow.shape

#TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
tfidf.shape

# Construcciond del Modelo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:1022,:]
test_bow = bow[1022:,:] 

# Dividiendo data entre training y test
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['target'],  
                                                          random_state=42, 
                                                          test_size=0.3,
                                                          train_size=0.7)

#Regresion Logistica

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

#Prediccion en base a las BOW
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) # calculating f1 score
#f1_score(yvalid, prediction_int,zero_division='warn') 
print("Prediccion en base a la BOW:    ", f1_score(yvalid, prediction_int), "\n")

test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['target'] = test_pred_int
submission = test[['target']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file

#Prediccion usando caracteristicas TF-IDF

train_tfidf = tfidf[:1542,:]
test_tfidf = tfidf[1542:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)
print("Prediccion en base a TF-IDF:    ", f1_score(yvalid, prediction_int), "\n")
'''