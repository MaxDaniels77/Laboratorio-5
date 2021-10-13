# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:52:47 2021

@author: cyber
"""

import numpy as np
import matplotlib.pyplot as plt 
import os
from scipy.signal import find_peaks
from scipy.special import factorial
from scipy.stats import poisson
import pandas as pd
from scipy.optimize import curve_fit
os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Conteo de fotones')
# Pantalla = np.loadtxt('Foton BE.csv', delimiter = ',')
# Pantalla = np.loadtxt('Foton BE.csv', delimiter = ',')
Pantalla = np.load(r'D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo 3v/Medicion1.npz')
for item in Pantalla.keys():
    print(item)

for item in Pantalla.values():
    print(item)
#%%
Pantalla = Pantalla['screens']
plt.plot(range(len(Pantalla)), Pantalla)
plt.title("Pulso")
plt.ylabel("Voltaje[V]")
plt.xlabel("Tiempo[s]")
# plt.xlim([0,0.010])
plt.grid(True)
#%%
escala = 1.25 # microsegundos
resolucion = escala/250*1e-6 # Consutla
tiempo = np.arange(0, resolucion*(2500*500), resolucion)
data=Pantalla

#%% Funciones

def la_ajustadora(funcion, x_data, y_data, y_error, parametros_iniciales, nombres_de_los_parametros):
    parametros_optimos, matriz_de_cov = curve_fit(funcion, x_data, y_data, sigma=y_error, p0 = parametros_iniciales)
    desvios_estandar = np.sqrt(np.diag(matriz_de_cov))
    datos = pd.DataFrame(index = nombres_de_los_parametros)
    datos['Valor'] = parametros_optimos
    datos['Error'] = desvios_estandar
    return datos

def rec(x, m, b):
    return m*x + b

def poiss(k, lamb):
    '''poisson function, parameter lamb is the fit parameter'''
    return poisson.pmf(k, lamb)

def poiss1(k, lamb, A):
    return A*poisson.pmf(k, lamb)

def BE(n, n_):
    return n_**n/(1 + n_)**(1 + n)

def gauss0(x, a, c):
    return a*np.exp(-(x)**2/(2*c**2))

def gauss(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))
    
def la_real(x, x1, x2, A, B):
    return A^2*poisson.pmf(x, x1) + B^2*x2**x/(1 + x2)**(1 + x)
    
def busca_picos(Intensidades, delta, umbral):
    """
    Parameters
    ----------
    Intensidades : array
        Es una lista de valores.
    delta : entero
        define el largo del intervalo en torno al valor para buscar el maximo.
    umbral : float
        valor minimo para ser considerado maximo.

    Returns
    -------
    array
        una lista con las ubicaciones de los maximos en intensidades.
    """
    ubi_picos = []
    for pixel, intensidad in enumerate(Intensidades):
        if len(Intensidades) - delta > pixel > delta:
            entorno = Intensidades[pixel-delta:pixel+delta]
            if intensidad >= max(entorno) >= umbral and intensidad != Intensidades[pixel-1]:
                ubi_picos.append(pixel)
    return np.array(ubi_picos)
#%%
alturas = []
t_a = []
delta = 2


ubi_picos1 = busca_picos(-data, delta, 0.00001 )
ubi_picos2 = busca_picos(data, delta, 0.00001 )

for index in ubi_picos1:
    alturas.append(data[index])
    t_a.append(tiempo[index])
    
for index in ubi_picos2:
    alturas.append(data[index])
    t_a.append(tiempo[index])    

print(len(alturas))
#%%
plt.figure()
plt.plot(tiempo, data)
# plt.plot(tiempo[0:len(tiempo)]*1e6, data[0:len(tiempo)]*1e3)
plt.plot(np.array(t_a[0:len(t_a)]), np.array(alturas[0:len(t_a)]), 'o', color = 'red')
plt.title("Deteccion de picos")
plt.ylabel("Voltaje[mV]")
plt.xlabel("Tiempo[us]")
# plt.xlim([0, 4])
# plt.ylim([-10, 2])
plt.grid(True)
#%% HISTOGRAMA DE LOS DATOS CRUDOS (NO PICOS)
plt.figure()
plt.hist(Pantalla*100,bins=200)
plt.grid(True)
plt.legend()
plt.xlabel('Alturas[mV]')
plt.ylabel('Apariciones')    
plt.title('Histograma de picos')
plt.yscale("log")
# plt.ylim([100, 100000])


#%%
mediciones = np.array(alturas)*1e3
# mediciones = data*1e3
DV = 0.08

bins = np.arange(-14, 4, DV) - DV/2 

plt.figure(figsize=(10,6))
plt.hist(mediciones, bins, color='#0504aa', alpha = 0.7, rwidth = 0.85, label = 'Mediciones' )
# plt.plot(r_plot, P_plot, label = 'Ajuste Gaussiano', color = 'red')
plt.grid(True)
plt.legend()
plt.xlabel('Alturas[mV]')
plt.ylabel('Apariciones')    
plt.title('Histograma de picos')
plt.yscale("log")
plt.ylim([100, 100000])
#%% VERSION 2.0 DEL GRAFICO PARA QUE SE PUEDA VER DONDE CORTA
# EL UMBRAL CON LOS PICOS GENERADOS POR LOS FOTONES

# Generamos una figura
plt.figure(1,figsize=(10,6))
plt.clf()
plt.rc('font', family='serif', size=13)
fig, ax = plt.subplots(2,1,num=1, sharex=True)
# 1 Muestro la referencia.
ax[0].plot( data[1500:2500]*1000,tiempo[1500:2500],linewidth=0.5,label='Recorte de medición')
ax[0].vlines(-2.5,tiempo[1500:2500].min(),tiempo[1500:2500].max(),ls="--",label='Threshold')
#ax[0].plot(t, Referencia_y, label='Referencia')
ax[0].set_ylabel('Tiempo [us]', fontsize=13)
ax[0].grid(True)
ax[0].legend()
ax[0].set_title('Histograma de datos', fontsize=16)
# 2 Muestro la señal
ax[1].hist(mediciones, bins, color='#0504aa', alpha = 0.7, rwidth = 0.85, label = 'Mediciones')
ax[1].set_ylabel('Apariciones', fontsize=13), ax[1].grid(True)
ax[1].set_yscale('log')
ax[1].set_ylim(100,1e5)
ax[1].set_xlabel('Alturas[mV]', fontsize=13)
ax[1].vlines(-2.5,100,1000000,ls="--",label='Threshold') , ax[1].legend()

# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()  
plt.savefig('histograma edit propio.png')
#%%
Torres00 = np.arange(-0.76, 1.32, DV)
Torres0 = np.zeros(len(Torres00))
for k, element in enumerate(Torres00):
    Torres0[k] = np.round(element, 5)
    
Cuentas0 = np.zeros(len(Torres0))

for lugar, voltaje in enumerate(Torres0):
    for element in mediciones:
        if voltaje - DV < element <= voltaje + DV:
            Cuentas0[lugar] += 1
 #%% Para ver si cuenta bien
plt.figure()    
plt.plot(Torres0, Cuentas0, '.')
# plt.yscale("log")
plt.grid(True)
#%%
parametros1 = la_ajustadora(gauss, Torres0, Cuentas0, None, [1000,0.1, 0.1], ["a", 'b', "c"])
a0 = parametros1["Valor"]["a"]
b0 = parametros1["Valor"]["b"]
c0 = parametros1["Valor"]["c"]

r_plot = np.linspace(-5.2, 5.2, 1000)
P_plot = gauss(r_plot, a0, b0, c0)

plt.figure()
plt.plot(Torres0, Cuentas0, 'o', label = 'Mediciones')
plt.plot(r_plot, P_plot, label = 'Ajuste Gaussiano')
plt.grid(True)
plt.legend()
plt.xlabel('Voltaje [mV]')
plt.ylabel('Cantidad')    
plt.title('Ruido electrico') 
# plt.yscale('log')
plt.ylim([10, 700000])