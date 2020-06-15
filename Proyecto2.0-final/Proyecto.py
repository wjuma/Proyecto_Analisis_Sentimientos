import csv
import re
import math
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import Datos_tweets as dt

#---------
from nltk.stem.snowball import SnowballStemmer
def funcion_curacion(palabras):
    tokens = [re.sub(r'[-()\"#/@;:´<>{}`+=~|.!?,’0-9,A¡“”…¡¡¡]', ' ', i.lower()).split() for i in palabras]
    stopW = stopwords.words('spanish')
    for h in tokens:
        #print(h)
        for m in h:
            if m in stopW:
                h.remove(m)
        for m in h:
            if m in stopW:
                h.remove(m)
    # print("\ntexto eliminado las StopWords:")
    # for i in tokens:
    #     print(i)
    spanishStemmer = SnowballStemmer("spanish", ignore_stopwords=True)
    stemmer2 = [[spanishStemmer.stem(m) for m in h] for h in tokens]
    # print("\nSteamming:")
    # for i in stemmer2:
    #      print(i)
    return stemmer2
def leerArchivo(path):
    archivo=[]
    filerR = open(path, 'r')
    for linea in filerR:
        # titulosR.append(linea)
        archivo.append(re.sub(r'\n', '', linea))
    return archivo

def funcion_curacionDiccionario(palabras):
    spanishStemmer = SnowballStemmer("spanish", ignore_stopwords=True)
    stemmer2 = [spanishStemmer.stem(m) for m in palabras]
    lista=[]
    for i in stemmer2:
        if i not in lista:
            lista.append(i)
    return lista

#-**********************************************************************************************************************
def pesado(n):
    if n > 0:
        return round(1 + math.log10(n), 2)
    else:
        return 0
def cacluloIDF(n, df):
    return round(math.log10(n/df), 3)

def funcionVocabulario(tweets):
    vocabularioTweet = []
    for tweet in tweets:
        for palabraTweet in tweet:
            if palabraTweet not in vocabularioTweet:
                vocabularioTweet.append(palabraTweet)
    return vocabularioTweet

def funcion_TF_IDF(documentosV):
    vocabulario = funcionVocabulario(documentosV)
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
porcentaje=[]
def funcion_coseno(idf_tf):
    N = len(idf_tf[0])
    modulos = []
    for i in range(len(idf_tf[0])):
        t = 0
        for j in range(len(idf_tf)):
            t += idf_tf[j][i] ** 2
        modulos.append(modulo(t))
    logNorm = [[resModulo(idf_tf[j][i], modulos[i]) for j in range(len(idf_tf))] for i in range(len(idf_tf[0]))]
    matrisT = [[cos(ks, kr) for ks in logNorm[-2:]] for kr in logNorm[:-2]]
    ResultadoSentimiento = ['{} => Positivo'.format(pos) if pos[0] > pos[1] else '{} => Negativo'.format(pos) for pos in matrisT]
    # print('Matriz Distancias Coseno')

    resultadoTotalCoseno = []
    pos=0
    neg=0
    for i in matrisT:
        if i[0]>i[1]:
            resultadoTotalCoseno.append('{} => Positivo'.format(i))
            #print(i,'-->Positivo')
            pos+=1
            continue
        if i[0]==i[1]:
            resultadoTotalCoseno.append('{} => Neutro'.format(i))
        else:
            neg = +1
            resultadoTotalCoseno.append('{} => Negativo'.format(i))
            #print(i, '-->Negativo')

    porcentaje.append((pos/len(matrisT)*100))
    porcentaje.append((neg/len(matrisT)*100))


    return resultadoTotalCoseno

def modulo(n):
    return math.sqrt(n)

def cos(v1, v2):
    return round(sum(v1[i] * v2[i] for i in range(len(v1))), 3)

def resModulo(v1, m):
    if m == 0:
        return 0
    else:
        return round(v1/m, 3)
#-**********************************************************************************************************************

def funcionJaccard(tweets,sentimiento):
    conjuntoTweets = [set(i) for i in tweets]
    conjuntoSentimiento = [set(i) for i in sentimiento]
    Resuldatos = []
    def jaccard(q, d):
        return round(len(q & d) / len(q | d), 4)
    for k in conjuntoTweets:
        respuestasJaccard = [jaccard(k, r) for r in conjuntoSentimiento]
        Resuldatos.append(respuestasJaccard)
    #ResultadoSentimiento = ['{} => Positivo'.format(pos) if pos[0] > pos[1] else '{} => Negativo'.format(pos) for pos in Resuldatos]
    # print('\n********************Distancia Jacard***********************')
    resultadoTotalJacard = []
    positivo=0
    neg=0
    for pos in Resuldatos:
        if pos[0]>pos[1]:
            #print(pos, ' => Positivo')
            positivo+=1
            resultadoTotalJacard.append('{} => Positivo'.format(pos))
            continue
        if pos[0]==pos[1]:
            #print(pos, ' => Neutro')
            resultadoTotalJacard.append('{} => Neutro'.format(pos))
        else:
            #print(pos, ' => Negativo')
            neg+=1
            resultadoTotalJacard.append('{} => Negativo'.format(pos))
    porcentaje.append((positivo/len(Resuldatos)*100))
    porcentaje.append((neg / len(Resuldatos) * 100))
    return resultadoTotalJacard

#-**********************************************************************************************************************

# filerR = open('./documentos/Coronavirus_Ecu_tweets.csv', 'r', encoding='utf-8')
# for i in filerR:
#     datos.append(re.sub(r'http\S+', '', i))
datos= []
curacion = []
sentimiento = []
result=[]

def consulta():
    #dt.buscar(nombre)
    lista = dt.lista[:]
    print('DEL TWEETER 2--->',lista)
    print('DEL TWEETER 2--->', len(lista))
    for j in lista:
        # print(j)
        datos.append(re.sub(r'http\S+', '', j))

    curacion.append(funcion_curacion(datos))
    path=['./documentos/Positivos2.csv', './documentos/Negativos2.csv']
    for i in path:
        sentimiento.append(funcion_curacionDiccionario(leerArchivo(i)))

    Matris=curacion[0]+sentimiento
    result.append(funcion_coseno(funcion_TF_IDF(Matris))[:])
    result.append(funcionJaccard(curacion[0], sentimiento)[:])
    print("Mostrar 2-->",curacion[0])
    print("Mostrar 2-->", len(curacion[0]))

def limpiaDatos():
    del datos[:]
def limpiaCuracion():
    del curacion[:]
    del sentimiento[:]
    del result[:]
    del porcentaje[:]
    #del resultadoTotalCoseno[:]
    #del resultadoTotalJacard[:]
# print('RESULTADOS')
# for i in range(len(datos)):
#     #print(i+1, ":", datos[i], " ", totalCoseno[i], " ", totalJaccard[i])
#     #print('{:>2}: {:<100} {} {:<28} {}'.format(i, ' '.join(curacion[0][i]), ' ', totalCoseno[i], totalJaccard[i]))
#     print('{:>2}: {:<100} {} {:<28} {}'.format(i,datos[i],' ',result[0][i],result[1][i]))
#-----------------------
# consulta('amor')
# for i in datos:
#     print("Datos",i)