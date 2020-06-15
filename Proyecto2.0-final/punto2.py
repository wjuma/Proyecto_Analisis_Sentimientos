import csv
import re
import math
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def funcion_curacion(palabras):
    tokens = [re.sub(r'[-()\"#/@;:´<>¿{}`+=~|.!?,’‘0-9,A¡“”…¡¡¡»]', ' ', i.lower()).split() for i in palabras]
    stopW = stopwords.words('spanish')
    for h in tokens:
        for m in h:
            if m in stopW:
                h.remove(m)
        for m in h:
            if m in stopW:
                h.remove(m)
        for m in h:
            if m in stopW:
                h.remove(m)

    spanishStemmer = SnowballStemmer("spanish", ignore_stopwords=True)
    stemmer2 = [[spanishStemmer.stem(m) for m in h] for h in tokens]
    # print("\nSteamming:")
    # for i in stemmer2:
    #      print(i)
    return stemmer2

def funcionVocabulario(tweets):
    vocabularioTweet = []
    for tweet in tweets:
        for palabraTweet in tweet:
            if palabraTweet not in vocabularioTweet:
                vocabularioTweet.append(palabraTweet)
    return vocabularioTweet

def pesado(n):
    if n > 0:
        return round(1 + math.log10(n), 2)
    else:
        return 0
def cacluloIDF(n, df):
    return round(math.log10(n/df), 3)

def funcion_TF_IDF(documentosV):
    vocabulario = funcionVocabulario(documentosV)
    #print('VOCA',len(vocabulario))
    N = len(documentosV)
    cuenta = []
    WTF = []
    idf = []
    dfl = []
    tf_idf = []
    for k in range(len(vocabulario)):
        pal = vocabulario[k]
        conteo = [tok.count(pal) for tok in documentosV]
        peso = [pesado(tok.count(pal)) for tok in documentosV]
        cuenta.append(conteo)
        WTF.append(peso)
        df = 0
        for i in peso:
            if i != 0:
                df += 1
        dfl.append(df)
        idf.append(cacluloIDF(N, df))
        opera_idf_tf = [e * cacluloIDF(N, df) for e in peso]
        tf_idf.append(opera_idf_tf)
    # print('{:<20} {:<20} {:<25} {:<8} {:<15} {}'.format('Palabra', 'Matriz TF', 'Matriz WTF', 'DF', 'IDF', 'Matriz TF-IDF'))
    # print('')
    # for pos in range(len(vocabulario)):
    #     print('{:<15} {} {} {} {} {:<7} {:<5} {} {:<10} {} {}'.format(vocabulario[pos], ':',cuenta[pos], '', WTF[pos], '', dfl[pos], '', idf[pos], '', tf_idf[pos]))
    return tf_idf




#----------------------------------------------------------------------------------------------------------------------------------
datos=[]
calificacion=[]
with open('./documentos/lista-posi-nega-nueva2.csv','r', newline='', encoding='utf8') as File:
    reader = csv.reader(File,delimiter=';')
    # reader = csv.reader(File)
    for row in reader:
        #print(row[2])
        datos.append(re.sub(r'http\S+', '', row[0]))
        if row[1]=='N':
            calificacion.append(0)
        else:
            calificacion.append(1)

# for i in range(10):
#     print(datos[i], calificacion[i])
#
#-----------------------------------------------------------------

curacion=[]
curacion.append(funcion_curacion(datos))
y=calificacion
valorx=funcion_TF_IDF(curacion[0])
arr1 = np.array(valorx)
x = arr1.transpose()
