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

# Pantalla = np.loadtxt('Foton BE.csv', delimiter = ',')
# Pantalla = np.loadtxt('Foton BE.csv', delimiter = ',')
Pantalla = np.load(r'D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo v3 dia 2 disco d/Grupo v3/1000 ventanas apagado 10mv 1us.npz')
prelim = np.loadtxt(r'D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo 3v/data_preliminar_100ohm 25ns.txt',delimiter=',')
 
for item in Pantalla.keys():
    print(item)

for item in Pantalla.values():
    print(item)
    
#%%

plt.figure(1,figsize=(10,6))
plt.clf()
plt.rc('font', family='serif', size=13)
fig, ax = plt.subplots(1,1,num=1, sharex=True)
# 1 Muestro la referencia.
ax.plot(prelim[0], prelim[1], label='Osc. screen')
ax.set_xlabel("Tiempo[s]")
ax.set_ylabel("Voltaje[V]")
ax.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
ax.set_title('Pulso 25 ns')
ax.legend(loc='lower right')
ax.grid(True)
ax.set_xlim(prelim[0].min(),prelim[0].max())
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()  
plt.savefig('pulso 25 ns.png')

#%% DATOS CRUDOS

Pantalla = Pantalla['screens']
plt.plot(range(len(Pantalla)), Pantalla)
plt.title("Pulso")
plt.ylabel("Voltaje[V]")
plt.xlabel("Tiempo[s]")
# plt.xlim([0,0.010])
plt.grid(True)

#%% HISTOGRAMA DE LOS DATOS CRUDOS DE LAS CUENTAS APAGADAS

plt.figure(1,figsize=(8,6))
plt.clf()
plt.rc('font', family='serif', size=13)
fig, ax = plt.subplots(1,1,num=1, sharex=True)
# 1 Muestro la referencia.
a = ax.hist(Pantalla*1e3 ,bins = 70, label='Histograma')
ax.set_xlabel("Alturas [mV]")
ax.set_ylabel("Apariciones")
ax.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
ax.set_title('Histograma de datos crudos')
# ax.set_yscale('log')
ax.legend()
ax.grid(True)
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()  
plt.savefig('Histograma de cuentas apagadas.png')

#%% AJUSTE DE GAUSSIANA DE CUENTAS OSCURAS

'''AL usar plt.hist como se configuro previamente es necesario matar los puntos que quedan
pegados en y = 0 y tomar los maximos correspondientes, para esto se realiza el siguiente 
crop en los datos'''

x = a[1][np.where(a[0]>0)]
y = a[0][np.where(a[0]>0)]
print(y.max())
# y = y/y.max() # normalice para que el algoritmo converja mejor

# plt.plot(x,y,".") # Chequeo que lo que obtuve sea correcto

parametros_optimos, matriz_de_cov = curve_fit(gauss, x, y, p0=[500000,0.6, 0.8])
error = np.sqrt(np.diag(matriz_de_cov))
X = np.linspace(x.min(),x.max(),1000)
plt.plot(x,y,".") 
plt.plot(X,gauss(X,*parametros_optimos))
print(f' Parametros = {parametros_optimos}\n errores={error}')

#%%

Pantalla = np.load(r'D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo v3 dia 2 disco d/Grupo v3/1000 ventanas encendido 10mv 1us.npz')

for item in Pantalla.keys():
    print(item)

for item in Pantalla.values():
    print(item)
    
data=Pantalla['screens']
escala = 1.25 # microsegundos
resolucion = escala/250*1e-6 # Consutla
tiempo = np.arange(0, resolucion*(len(data)), resolucion)

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
#%% ALTURAS RECORTANDO LOS PICOS
t = np.array(t_a[0:len(t_a)])
peak = np.array(alturas[0:len(t_a)])*1e2
# peak_cut_index = np.where(peak<-) 
plt.plot(t,peak)
#%% HISTOGRAMA DE LOS DATOS CRUDOS (NO PICOS)

plt.figure()
plt.hist(data*100,bins=200)
plt.grid(True)
plt.legend()
plt.xlabel('Alturas[mV]')
plt.ylabel('Apariciones')    
plt.title('Histograma de los datos')
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
ax[0].set_ylabel('Tiempo [$\mu s$]', fontsize=13)
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




#%% RECORTANDO EN EL UMBRAL FIJADO A OJO EN -2.5
#    recortando en el umbralllll

data_escalada = data*1e3
index_recorte = np.where(data_escalada<-2.5)
# plt.hist(data_escalada[index_recorte],bins=100)
data = data[index_recorte]
tiempo = tiempo[index_recorte]

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
ax[0].set_ylabel('Tiempo [$\mu s$]', fontsize=13)
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



'''Puedeb estar faltando cosas aca''' 


#%% Estudio de la correlacion, para cuando gira el plato mediciones en la escala del tiempo de correlacion
# sin resistencia conectada para usarlo de integrador

os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Conteo de fotones\Grupo v3 dia 2 disco d\Grupo v3')
# 5s

# DatosC = np.loadtxt('D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo v3 dia 2 disco d/Grupo v3/1 ventana determinacion de Tc, 5v,5s.csv', delimiter = ',')
# t = DatosC[0][255:2350] # recorte para el set de datos 5s  
# d = DatosC[1][255:2350]

#250 ms

# DatosC = np.loadtxt('D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo v3 dia 2 disco d/Grupo v3/1 ventana determinacion de Tc, 5v,250ms.csv', delimiter = ',')
# t = DatosC[0][:2190] # recorte set de datos 250 ms
# d = DatosC[1][:2190]

#500 ms 

DatosC = np.loadtxt('D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo v3 dia 2 disco d/Grupo v3/1 ventana determinacion de Tc, 5v,500ms.csv', delimiter = ',')
t = DatosC[0][20:2250] # recorte set de datos 
d = DatosC[1][20:2250]


# DatosC = np.load(r'D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo v3 dia 2 disco d/Grupo v3/1 ventana determinacion de Tc, 5v,500ms.npz')


#%%

plt.plot(t, d)
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
plt.title("Pantalla")

#%%

prom = np.mean(d)
# prom = 0
Corre = np.correlate(d - prom, d - prom, mode = "full")
print(len(Corre))

#%%

paso = t[1]-t[0]
ti_c = np.linspace(0, paso*2*len(t)-2, 2*len(t)-1)
plt.plot(ti_c, Corre)
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Correlacion')
plt.title('Autocorrelacion' )

#%% Busca Tc, busca el ancho a mitad de altura 

def busca_Tc(paso, correlacion):
    L = len(correlacion)
    M = max(correlacion)
    for i in range(L):
        if correlacion[i] > M/2:
            i1 = i
            break
    for i in range(L):
        if correlacion[-i] > M/2:
            i2 = L - i + 1    
            break
    Tc = (i2 - i1)*paso
    return Tc

#%%
corr_calc = busca_Tc(paso,Corre)
print(f'La correlacion es: {corr_calc}')


#%%

Tcs = []
Tt = 51
paso = 0.001/250

for i in range(Tt):
    d0 = d[i*2500: (i+1)*2500] - np.mean(d[i*2500: (i+1)*2500])
    Corre0 = np.correlate(d0[:2500], d0[:2500], mode = "full")
    Tc0 = busca_Tc(paso, Corre0) 
    Tcs.append(Tc0)

print('Tc = ({:4.2f} ± {:4.2f})s'.format(np.mean(Tcs)*1000, np.std(Tcs)*1000))

























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