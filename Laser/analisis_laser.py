# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:53:46 2021

@author: cyber
"""
# %% IMPORTS
import scipy.stats as stats
import scipy.optimize as opt
import scipy.misc
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.special import factorial
from scipy.stats import poisson
import pandas as pd
from scipy.optimize import curve_fit
import imageio
import cv2
# os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Laser')
# %% PLOTEO DE CURVAS
# os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Laser\Mediciones') # windows
os.chdir(
    "/media/daniel/OS/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Laser/Mediciones"
)  # linux

prelim = np.loadtxt(
    r"Eficiencia de la cavidad lineal P vs I.csv", delimiter=",", skiprows=1)

I = prelim[:, 0]
P = prelim[:, 1]
error_y = prelim[:, 2]*3
plt.figure(1, figsize=(8, 4))
plt.clf()
plt.rc("font", family="serif", size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
# 1 Muestro la referencia.
ax.errorbar(I, P, error_y, fmt="+", label="Mediciones")
ax.set_xlabel("Corriente [A]")
ax.set_ylabel("Potencia [mW]")
ax.ticklabel_format(axis="y", style="sci", scilimits=(1, 4))
ax.set_title("Curva P vs I")
ax.legend(loc="lower right")
ax.grid(True)
# ax.set_xlim(prelim[0].min(), prelim[0].max())
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()
# plt.savefig('..\Curva P vs I.png') # windows
plt.savefig("../Curva P vs I.png")  # linux
#%% AJUSTE LINEAL
def lineal(x, a, b): return a * x + b

x = I[1:]
y = P[1:]

error_y = error_y[1:]*4
parametros_optimos, matriz_de_cov = curve_fit(lineal, x, y, sigma=error_y)
error = np.sqrt(np.diag(matriz_de_cov))

plt.errorbar(x, y, yerr=np.sqrt(y), fmt="+", label="Datos")
plt.plot(x, lineal(x, *parametros_optimos), ".", label="Ajuste")
# plt.yscale("log")
print(f" Parametros = {parametros_optimos}\n errores={error}")
plt.title("Ajuste Bose-Einstein")
# Analisis del estadistico χ²
# esto es: grados de libertad = #mediciones - #Parámetros - 1
grados_lib = len(y) - len(parametros_optimos) - 1
modelo = lineal(x, *parametros_optimos)
chi_cuadrado = np.sum(((y - modelo) / error_y) ** 2)  # Cálculo del χ²
# Cálculo de su p-valor asociado
p_chi = 1 - chi2.cdf(chi_cuadrado, grados_lib)
# Printeamos los resultados
print(f"chi^2: {chi_cuadrado}")
print(f"Grados de Libertad: {grados_lib}")
print(f"p-valor del chi^2: {p_chi:.3g}")
if p_chi < 0.05:
    print("Se rechaza la hipótesis de que el modelo ajuste a los datos.\n")
else:
    print("No se rechaza la hipótesis de que el modelo ajuste a los datos.\n")
#%% Funciones para modo tem
'''La idea es hacer que el laser luego de pegar sobre una lente, para amplificar,
pegue sobre una hoja milimetrada o algo por el estilo para poder caracterizar el modo
Para eso es la funcion calibrate scale'''

def gauss(x, a, b, c): # para el ajuste
    return a*np.exp(-(x-b)**2/(2*c**2))

def diff(v):
    v_diff = []
    for i in range(1, len(v) - 1):
        v_diff.append(v[i][0] - v[i - 1][0])
        
    return v_diff

#funcion para calibrar la escala del calibre. Toma un vector y devuelve el promedio de las diferencias entre elementos de ese vector
#junto con su desviación estandar
def calibrateScale(x):
    v_diff = []      
    v_diff = diff(x)
    mm = np.mean(v_diff)
    mm_err = 2*np.std(v_diff)
    return mm, mm_err


#kernel en el codigo de Richi (de donde sale este kernel? Porque usando este kernel la convolucion es tan ruidosa?)
s = [[1, 2, 1],  
     [0, 0, 0], 
     [-1, -2, -1]]
#%% CARGA DE LA IMAGEN A ANALIZAR

path = 'Modo Tem00.jpeg'
#cargamos la imagen a analizar
# AA = imageio.imread(path)
A = cv2.imread(path,0)
#layer de la imagen como matriz con valores de 0-255
# A = AA[:,:,1]# tomo un canal que no sature

#grafica la imagen original


image = cv2.line(A,(0,915),(1200,915),(255,0,0),2)
image = cv2. resize(image, (600, 800)) # Resize image.
window_name='linea'
cv2.imshow(window_name,image)

#%% TOMA DEL PERFIL Y AJUSTE 

perf = image[459,:] # El perfil se tomo a ojo (al menos para el modo gaussiano)

plt.plot(range(len(perf)),perf)


#%%
y = perf

x = range(len(perf))

parametros_optimos, matriz_de_cov = curve_fit(
    gauss, x, y, sigma=np.sqrt(y)+1, p0=[500, 629,100])
error = np.sqrt(np.diag(matriz_de_cov))

plt.errorbar(x, y, yerr=np.sqrt(y), fmt='+', label='Datos')
plt.plot(x, gauss(x, *parametros_optimos), ".", label='Ajuste')
# plt.yscale('log')
print(f' Parametros = {parametros_optimos}\n errores={error}')
plt.title('Ajuste Gaussiano')
# Analisis del estadistico χ²
# esto es: grados de libertad = #mediciones - #Parámetros - 1
grados_lib = len(y) - len(parametros_optimos)-1
modelo = gauss(x, *parametros_optimos)
error_y = np.sqrt(y)
chi_cuadrado = np.sum(((y - modelo) / error_y)**2)  # Cálculo del χ²
# Cálculo de su p-valor asociado
p_chi = 1 - chi2.cdf(chi_cuadrado, grados_lib)
# Printeamos los resultados
print(f"chi^2: {chi_cuadrado}")
print(f"Grados de Libertad: {grados_lib}")
print(f"p-valor del chi^2: {p_chi:.3g}")
if p_chi < 0.05:
    print("Se rechaza la hipótesis de que el modelo ajuste a los datos.\n")
else:
    print("No se rechaza la hipótesis de que el modelo ajuste a los datos.\n")