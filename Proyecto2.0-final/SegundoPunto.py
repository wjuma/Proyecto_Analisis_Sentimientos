import pandas as pd
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk # for text manipulation
import warnings

datos = []
calificacion = []
with open('./documentos/Archivo.csv', 'r', newline='', encoding='utf8') as File:
    reader = csv.reader(File, delimiter=';')
    for row in reader:
        #print(row)
        datos.append(row)
        # if row[1] == 'N':
        #     calificacion.append(0)
        # else:
        #     calificacion.append(1)

print(datos)
sel1 = random.choice(datos)
print(sel1)
