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

# os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Laser')
# %% PLOTEO DE CURVAS
# os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Laser\Mediciones') # windows
os.chdir(
    "/media/daniel/OS/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Laser/Mediciones"
)  # linux

prelim = np.loadtxt(
    r"Eficiencia de la cavidad lineal P vs I.csv", delimiter=";", skiprows=1
)

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
