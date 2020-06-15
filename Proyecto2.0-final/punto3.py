from builtins import print
from textblob import TextBlob
import re
import Datos_tweets as dt

a = []
b = []
lis = []
valor = []
calculo=[]
valtotal=[]
valtotal2=[]
def procesar ():
    #dt.buscar(nombre)
    lista = dt.lista[:]
    print("DEL TWEETER 1 ---->",lista)
    print("DEL TWEETER 1--->", len(lista))

    #print("nuevovovov",lista)

    # elimino los https
    for j in lista:
        # print(j)
        b.append(re.sub(r'http\S+', '', j))
    # print("sin https", a)

    for j in b:
        # print(j)
        a.append(re.sub(r'\n+', '', j))
    # print("sin https", a)

    # print("lista\n")
    # for s in a:
    #     print(s)

    # comoensamos el analisis con el textblob
    pos_count = 0
    pos_correct = 0
    neg_count = 0
    neg_correct = 0

    # for line in a:
    #     analysis = TextBlob(line)
    #     try:
    #         eng = analysis.translate(from_lang='es', to='en')
    #         print("VALIO---->",eng.sentiment.polarity)
    #         valor.append(eng.sentiment.polarity)
    #         if eng.sentiment.polarity > 0:
    #             lis.append("positivo:" + line)
    #             print("positivo:\n", line)
    #             pos_correct += 1
    #         pos_count += 1
    #         if eng.sentiment.polarity <= 0:
    #             lis.append("negativo:" + line)
    #             print("negativo:\n", line)
    #             neg_correct += 1
    #         neg_count += 1
    #     except:
    #         print("El elemento no está presente")
    print("Lista A:",a)
    print("Lista A:", len(a))
    for line in a:
        analysis = TextBlob(line)
        try:
            eng = analysis.translate(to='en')
            valor.append(eng.sentiment.polarity)
            if eng.sentiment.polarity > 0:
                lis.append("positivo:" + line)
                #print("positivo:\n", line)
                pos_correct += 1
            pos_count += 1
            if eng.sentiment.polarity <= 0:
                lis.append("negativo:" + line)
                #print("negativo:\n", line)
                neg_correct += 1
            neg_count += 1
        except:
            print("El elemento no está presente")

    #print("Precisión positiva = {}% con {} tweets".format(pos_correct / pos_count * 100.0, pos_count))
    #print("Precisión negativa = {}% con {} tweets".format(neg_correct / neg_count * 100.0, neg_count))
    print("Muestra 1------>",lis)
    print("Muestra 1------>",len(lis))

    valtotal.append(pos_correct / pos_count * 100.0)
    valtotal2.append(neg_correct / neg_count * 100.0)

def limpiaLis():
    del lis[:]

def limpiaA():
    del a[:]

def limpiaB():
    del b[:]
    del valtotal[:]
    del valtotal2[:]

'''
    calcularPos = 0
    calcularNeg = 0
    if pos_count == 0:
        calcularPos = 0
    else:
        calcularPos = (pos_correct / pos_count * 100.0)
    if neg_count == 0:
        calcularNeg = 0
    else:
        calcularNeg(neg_correct / neg_count * 100.0)
    calculo.append(calcularPos)
    calculo.append(calcularNeg)

'''


