# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 20:14:07 2021

á é í ó ú ñ

@author: Pc
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.special import factorial
from scipy.stats import poisson
import pandas as pd
from scipy.optimize import curve_fit

plt.rcParams['figure.figsize'] = [16,8]
plt.rcParams['font.size'] = 25
plt.rcParams['font.family'] = 'times new roman'

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


"""
No conseguimos buenos resultados con np.find_peaks() de numpy, asi que creamos una
prueben esa tambien 
"""
#%%
Pantalla = np.loadtxt('Foton BE.csv', delimiter = ',')

plt.plot(Pantalla[:2500,0], Pantalla[:2500,1])
plt.title("Pulso")
plt.ylabel("Voltaje[V]")
plt.xlabel("Tiempo[s]")
# plt.xlim([0,0.010])
plt.grid(True)

#%% Carga de datos
# Datos = np.loadtxt('Muestreo dia 2 - 1us pro.csv', delimiter = ',')
Datos = np.loadtxt('Muestreo dia 3 BE - 1kO - 2.5us.csv', delimiter = ',')

escala = 2.5
resolucion = escala/250*1e-6
tiempo = np.arange(0, resolucion*(2500*1001), resolucion)
data = Datos[:,1]

#%% Para recortar decimales "ficticios" dependiendo el bin que se elija puede ser necesario - 1.2*1e-3
# paso = 2.5/250*1e-6
# # tiempo = Datos[:,0]
# tiempo = np.arange(0, paso*(2500*1001), paso)
# data0 = Datos[:,1]
# data = np.zeros(len(data0))

# for k, element in enumerate(data0):
#     data[k] = np.round(element, 5)
    

#%% Busca picos nuestro
alturas = []
t_a = []
delta = 8

ubi_picos1 = busca_picos(-data, delta, 0.00001 )
ubi_picos2 = busca_picos(data, delta, 0.00001 )

for index in ubi_picos1:
    alturas.append(data[index])
    t_a.append(tiempo[index])
    
for index in ubi_picos2:
    alturas.append(data[index])
    t_a.append(tiempo[index])    

print(len(alturas))
#%% Ploteo de datos y picos, para ver si estan bieseteados los parametros de busqueda

plt.figure()
plt.plot(tiempo[0:len(tiempo)]*1e6, data[0:len(tiempo)]*1e3)
plt.plot(np.array(t_a[0:len(t_a)])*1e6, np.array(alturas[0:len(t_a)])*1e3, 'o', color = 'red')
plt.title("Deteccion de picos")
plt.ylabel("Voltaje[mV]")
plt.xlabel("Tiempo[us]")
plt.xlim([0, 4])
plt.ylim([-10, 2])
plt.grid(True)

#%% Para ver la discretizacion si no se sabe la escala de voltajes empleada para la medicion

# plt.plot(ubi_picos1, alturas[:7148], '.')
# plt.title("Discretizacion")
# plt.ylabel("Voltaje[mV]")
# plt.xlabel("Posicion")
# plt.ylim([-0.003, -0.002])
# # plt.ylim([0, 1000])
# plt.grid(True)

#%% Plot histograma
    
mediciones = np.array(alturas)*1e3
# mediciones = data*1e3
DV = 0.08

bins = np.arange(-14, 4, DV) - DV/2 

plt.figure()
plt.hist(mediciones, bins, color='#0504aa', alpha = 0.7, rwidth = 0.85, label = 'Mediciones' )
# plt.plot(r_plot, P_plot, label = 'Ajuste Gaussiano', color = 'red')
plt.grid(True)
plt.legend()
plt.xlabel('Alturas[mV]')
plt.ylabel('Apariciones')    
plt.title('Histograma de picos')
plt.yscale("log")
plt.ylim([100, 100000])

#%% Pasaje de los datos a histograma (contar cuantos caen en cada columna)
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

#%% Para ajustar la campana del fondo electrico

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

#%% Cuenta picos
"""
Aca arranca la estadistica de Poisson o BE,
la idea es que busca los picos que hay en cada pantalla (2500)
en cada una de las Tt pantallas, que superren el umbral. Despues divide
la pantalla en fraccion partes y asigna a cada una la cantidad de picos 
que se detectaron ahi, eso es N1_0, que finalmente se concatena con N1 que 
tiene las apariciones de picos en cada ventana. 
"""

N1 = []

fraccion = 5

Tt = 1001
umbral1 = 4
distancia = 8

for i in range(Tt):
    print(1000-i)
    d = -data[i*2500: (i+1)*2500]

    picos1 = busca_picos(d, distancia, umbral1*1e-3)
    N1_0 = np.zeros(int(fraccion))
    for i in range(int(fraccion)):
        for pic in picos1:
            if i*(2500//fraccion) - 1 < pic <= (i + 1)*(2500//fraccion) - 1 <= 2500:
                N1_0[i] += 1  
    N1 = N1 + N1_0.tolist() 

#%% Histograma de N1, junto con sus posibles ajustes, comentados

bins = np.arange(-0.5, 15.5, 1)

plt.figure()
plt.hist(N1, bins, color='#0504aa', alpha = 0.7, rwidth = 0.85, label = 'Mediciones' )
# plt.plot(Torres, P_plot2*Mod, 'o', markersize = 10, color = 'red', label = 'Ajuste Poissoniano')
# plt.plot(Torres, P_plot1*Mod, 'o', markersize = 10, color = 'black', label = 'Ajuste BE')
plt.grid(True)
plt.legend()
plt.xlabel('Número de fotocuentas')
plt.ylabel('Apariciones')    
plt.title('Histograma - ventana: 3.3us')

#%% Contador, cuenta cuantas veces aparecio cada cantidad de picos
Torres = np.arange(0, 15, 1)
Cuentas = np.zeros(len(Torres))

for entero in Torres:
    for element in N1:
        if element == entero:
            Cuentas[int(entero)] += 1
 
Mod = sum(Cuentas)    
P = Cuentas/Mod
    
#%%
plt.figure()
plt.plot(Torres, P, 'o')
plt.grid(True)
# plt.legend()
plt.xlabel('Cantidad')
plt.ylabel('Apariciones')    
plt.title('Histograma') 

#%% Ajustes de las distribuciones, tambien se puede calcular l0 y plotear
parametros1 = la_ajustadora(BE, Torres, P, None, [0.1], ["lambda"])
parametros2 = la_ajustadora(poiss, Torres, P, None, [0.1], ["lambda"])

l0 = np.sum(Torres*P)
l1 = parametros1["Valor"]["lambda"]
l2 = parametros2["Valor"]["lambda"]

print(l0, l1, l2)

P_plot1 = BE(Torres, l0)
P_plot2 = poiss(Torres, l0)
P_plot3 = la_real(Torres, parametros3["Valor"]["x1"], parametros3["Valor"]["x2"], parametros3["Valor"]["A"], parametros3["Valor"]["B"])

plt.figure()
plt.plot(Torres, P_plot1, 'o', label = 'Ajuste Boseeinstiano')
plt.plot(Torres, P_plot2, 'o', label = 'Ajuste Poissoniano')
# plt.hist(N1, bins, color='#0504aa', alpha = 0.7, rwidth = 0.85, label = 'Mediciones' )
# plt.plot(Torres, P_plot3, 'o', label = 'Ajuste Real')

plt.plot(Torres, P, 'o', markersize = 10,  label = 'Mediciones')
plt.grid(True)
plt.legend()
plt.xlabel('Numero de fotones')
plt.ylabel('Frecuencia')    
plt.title('Histograma') 


#%% Para la estadistica de cero fotones / un foton en funcion de la ventana
# no nos salio, necesita los parametros bien calibrados, ventana chica con bastantes fotones

# Tt = 1001
# umbral1 = 4
# distancia = 8

# Fracciones = np.linspace(100, 1000, 10)
# Tiempo = 2500//Fracciones*0.01

# ceros1 = []
# unos1 = []

# for fraccion in Fracciones:
#     print(fraccion)
#     N1 = []
#     for i in range(Tt):
#         # print(1000-i)
#         d = -data[i*2500: (i+1)*2500]

#         picos1 = busca_picos(d, distancia, umbral1*1e-3)
#         N1_0 = np.zeros(int(fraccion))
#         for i in range(int(fraccion)):
#             for pic in picos1:
#                 if i*int(2500//fraccion) - 1 < pic <= (i + 1)*int(2500//fraccion) - 1 <= 2500:
#                     N1_0[i] += 1  
#         N1 = N1 + N1_0.tolist()            
#     Torres = np.arange(0, 2, 1)
#     Cuentas = np.zeros(len(Torres))

#     for entero in Torres:
#         for element in N1:
#             if element == entero:
#                 Cuentas[int(entero)] += 1
#     ceros1.append(Cuentas[0])            
#     unos1.append(Cuentas[1])    

#%%

# plt.figure()
# # plt.plot(Tiempo, ceros1, 'o', label = 'Ceros')
# plt.plot(Tiempo, unos1, 'o', label = 'Unos')
# plt.grid(True)
# plt.legend()
# plt.xlabel('Numero de fotones')
# # plt.ylabel('Frecuencia')    
# plt.title('Cero y Uno') 

#%% Giro para el giro del plato

DatosG = np.loadtxt('Giro 1.csv', delimiter = ',')
tiempoG = DatosG[:,0]
VG = -DatosG[:,1]
#%%

plt.plot(tiempoG, VG)
plt.plot(flancos, Vs, 'o', color = 'red')
plt.grid(True)
# plt.legend()
plt.xlabel('t')
plt.ylabel('V[V]')    
plt.title('Giro') 
    
#%%
lugares = []
flancos = []
Vs = []

for lugar in range(2500):
    if VG[lugar] - VG[lugar - 1] > 2:
        lugares.append(lugar)
        flancos.append(tiempoG[lugar])
        Vs.append(VG[lugar])


#%% Estudio de la correlacion, para cuando gira el plato mediciones en la escala del tiempo de correlacion
# sin resistencia conectada para usarlo de integrador
DatosC = np.loadtxt('Corr 1ms.csv', delimiter = ',')
t = DatosC[:,0]
d = DatosC[:,1]

#%%
plt.plot(t[2500*3:2500*4], d[2500*3:2500*4])
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
plt.title("Pantalla")

#%%
prom = np.mean(d[2500*3:2500*4])
Corre = np.correlate(d[2500*3:2500*4] - prom, d[2500*3:2500*4] - prom, mode = "full")
print(len(Corre))
#%%

paso = 0.001/250
ti_c = np.linspace(0, paso*4998, 4999)
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

Tcs = []
Tt = 51
paso = 0.001/250

for i in range(Tt):
    d0 = d[i*2500: (i+1)*2500] - np.mean(d[i*2500: (i+1)*2500])
    Corre0 = np.correlate(d0[:2500], d0[:2500], mode = "full")
    Tc0 = busca_Tc(paso, Corre0) 
    Tcs.append(Tc0)

print('Tc = ({:4.2f} ± {:4.2f})s'.format(np.mean(Tcs)*1000, np.std(Tcs)*1000))


#%% Correlacion con imagenes y demas, esto ya no es muy util quiza porqur no pudimos
# automatizar el analisis de imagenes para sacar el tiempo medio

# Cargamos la librerías que vamos a necesitar
import imageio
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
plt.rcParams['font.size'] = 25
plt.rcParams['figure.figsize'] = [10,10]

#%% Ejemplo: Aplicación de filtro para detección de bordes
imagen    = imageio.imread('IMG_2.jpg')

#%%
imagenc = imagen[1000:3000, 1500:3500, :]
imagenc2 = imagenc[500:1400, 700:1600, :]

plt.imshow( imagenc2 )
plt.title('Imagen Original')

#%%

img_R = imagenc2[:,:,0]
img_G = imagenc2[:,:,1]
img_B = imagenc2[:,:,2]

plt.figure(figsize = (7, 6))
plt.imshow( img_R )
plt.title('Canal R')
#%%
plt.figure(figsize = (7, 6))
plt.imshow( img_G )
plt.title('Speckle')
plt.xlabel("Píxeles")
plt.ylabel("Píxeles")
# plt.figure(figsize = (7, 6))
# plt.imshow( img_B )
# plt.title('Canal B')

#%% Ejemplo: Aplicamos filtro de detección de  bordes


# kernel del filtro
s = [[1, 2, 1],  
     [0, 0, 0], 
     [-1, -2, -1]]


# Calcula la convolucion de la imagen con el kernel (H aplica el filtro en la dirección x y V en la dirección y)
HR = signal.convolve2d(img_G, s)
VR = signal.convolve2d(img_G, np.transpose(s))
img_bordes = (HR**2 + VR**2)**0.5

plt.imshow( img_bordes )
plt.title('Bordes')
plt.xlabel("Píxeles")
plt.ylabel("Píxeles")

#%%
indices_mayores_a_55 = (img_bordes > 40)

img_binarizada80 = np.zeros( img_bordes.shape )

img_binarizada80[indices_mayores_a_55  ] = 1

plt.imshow( img_binarizada80 )
plt.title('Bordes')
plt.xlabel("Píxeles")
plt.ylabel("Píxeles")


