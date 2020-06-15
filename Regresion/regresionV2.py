import re # for regular expressions
import pandas as pd 
pd.set_option("display.max_colwidth", 200)
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk # for text manipulation
import warnings 
import csv
import re
import math
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

warnings.filterwarnings("ignore", category=DeprecationWarning)

columnas1 = ['text','numeral','target']
train = pd.read_csv('lista_ultima.csv', encoding = "latin-1", names=columnas1)
#train = train.drop(train[train['target']==1].index) # eliman los NP

train =train.replace(2, 0)

train


'''
columnas2 = ['id','text','target']
test = pd.read_csv('test.csv',encoding = "latin-1", names=columnas2)
test = test.drop(test[test['target']==1].index) # eliman los NP

pruebT = train[train['target'] == 1].head(10) # ayuda aver la cabezara de los 10 primero que calificamos como postivos 
'''
'''
#train.shape, test.shape
Out[41]: ((1022, 3), (520, 3))

'''

# train['target'].value_counts()



'''
2    527
1    494
0      1
Name: target, dtype: int64
'''
'''
length_train = train['text'].str.len()
length_test = test['text'].str.len()
# la distrubucion de la logitud de los tweets es mas o menos la misma para ambos casos para los datos de train y test 
plt.hist(length_train, bins=20, label="train_tweets")
plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
plt.show()

#limpieza

combi = train.append(test, ignore_index=True)
combi.shape

#(1542, 3)
# 


def remove_pattern(input_txt, pattern): # remueve patrones no deseados en el tweets
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt



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




# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['target'] == 1])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['target'] == 2])

# unnesting list
HT_regular = sum(HT_regular,[]) #com ams impacto dataset
HT_negative = sum(HT_negative,[])


a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags     
d = d.nlargest(columns="Count", n = 20)  #### hasthtag tus postivos .....
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

'''




