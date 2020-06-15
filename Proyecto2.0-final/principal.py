from builtins import print

from flask import Flask, render_template
from flask import request
import punto3 as p3
import Proyecto as p1
import Datos_tweets as dt
app = Flask(__name__)
@app.route('/',methods=['GET', 'POST'])

def raiz():
    palabraBuscar = ""
    cantidadT=0
    if request.method == 'POST':
        palabraBuscar = request.form['palabra']
        cantidadT=request.form['cantidad']
    busquedaPalabra = palabraBuscar + ""
    cantidad=int(cantidadT)
    print("PALABRA 1: ",busquedaPalabra)
    print("Cantidad: ",cantidad)

    if busquedaPalabra=='':
        print('No hay texto')
    else:
        dt.limpia()
        dt.buscar(busquedaPalabra,cantidad)
    return render_template('index.html',tweets=dt.lista[:],tamanio=len(dt.lista[:]))

@app.route('/Primero')
def primera():
    p3.limpiaB()
    p3.limpiaA()
    p3.limpiaLis()
    p3.procesar()
    return render_template('home.html',
                           listaTexto=p3.lis,
                           listaValor=p3.valor,
                           numTweets=len(p3.lis),
                           valorPositivo=p3.valtotal,
                           valorNegativo=p3.valtotal2)
    #return render_template("index.html")

@app.route('/Segundo')
def segunda():
    p1.limpiaDatos()
    p1.limpiaCuracion()
    p1.consulta()
    return render_template('parte1.html',
                               tex=dt.lista[:],
                               rescos=p1.result[0],
                               num=len(p1.result[0]),
                               resjacart=p1.result[1],
                           cosenoPorcentajePositivo=p1.porcentaje[0],
                           cosenoPorcentajeNegativo=p1.porcentaje[1],
                           jaccardPorcentajePositivo=p1.porcentaje[2],
                           jaccardPorcentajeNegativo=p1.porcentaje[3])
#    return render_template("example.html",mensa='Aque esta el mensaje:: {}'.format(busqueda))

if __name__== '__main__':
    app.run(debug=True)
