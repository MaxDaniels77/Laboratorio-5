# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:53:46 2021

@author: cyber
"""
# %% IMPORTS
from scipy.optimize import minimize
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
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
from scipy import special

# os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Laser')
# %% PLOTEO DE CURVAS
# windows
os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Laser\Mediciones')
# os.chdir(
#     "/media/daniel/OS/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Laser/Mediciones"
# )  # linux

prelim = np.loadtxt(
    r"Eficiencia de la cavidad lineal P vs I.csv", delimiter=",", skiprows=1
)

I = prelim[:, 0]
P = prelim[:, 1]/1000
error_y = prelim[:, 2]/1000
plt.figure(1, figsize=(8, 4))
plt.clf()
plt.rc("font", family="serif", size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
# 1 Muestro la referencia.
ax.errorbar(I, P, error_y, fmt="+", label="Mediciones")
ax.set_xlabel("Corriente [A]")
ax.set_ylabel("Potencia [W]")
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
# %% AJUSTE LINEAL


def lineal(x, a, b):
    return a * x + b


x = I[4:].copy()
y = P[4:].copy()
error_y_aj = error_y[4:].copy()

# %%
parametros_optimos, matriz_de_cov = curve_fit(lineal, x, y, sigma=error_y_aj)
error = np.sqrt(np.diag(matriz_de_cov))


# plt.errorbar(x, y, yerr=np.sqrt(y), fmt="+", label="Datos")
# plt.plot(x, lineal(x, *parametros_optimos), ".", label="Ajuste")

plt.figure(1, figsize=(8, 4))
plt.clf()
plt.rc("font", family="serif", size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
plt.rcParams['axes.facecolor'] = '#ffffff'
fig.set_facecolor('#ffffff')
# 1 Muestro la referencia.
ax.errorbar(I, P, error_y, fmt="+", label="Mediciones")
ax.plot(x, lineal(x, *parametros_optimos), "-", label="Ajuste")
ax.set_xlabel("Corriente [A]")
ax.set_ylabel("Potencia [W]")
ax.ticklabel_format(axis="y", style="sci", scilimits=(1, 4))
ax.set_title("Curva P vs I")
ax.legend(loc="upper left")
# ax.grid(True)
# ax.set_xlim(prelim[0].min(), prelim[0].max())
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.subplots_adjust(bottom=0.137, left=0.065, right=0.996, top=0.923)

plt.show()
# plt.savefig('..\Curva P vs I.png') # windows
plt.savefig("../Ajuste Curva P vs I.png")  # linux


# plt.yscale("log")
print(f" Parametros = {parametros_optimos}\n errores={error}")
plt.title("Curva de eficiencia cavidad lineal")
# Analisis del estadistico χ²
# esto es: grados de libertad = #mediciones - #Parámetros - 1
grados_lib = len(y) - len(parametros_optimos) - 1
modelo = lineal(x, *parametros_optimos)
chi_cuadrado = np.sum(((y - modelo) / error_y_aj) ** 2)  # Cálculo del χ²
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
    # %% Funciones para modo tem
"""La idea es hacer que el laser luego de pegar sobre una lente, para amplificar,
pegue sobre una hoja milimetrada o algo por el estilo para poder caracterizar el modo
Para eso es la funcion calibrate scale"""


def TEMX01(x, x0, I, w):
    Hp = special.hermite(1, monic=False)
    U = Hp((np.sqrt(2) * np.array(x - x0)) / w) * np.exp(
        -(np.array(x - x0) ** 2) / w ** 2
    )
    return I * U ** 2


def gauss(x, a, b, c):  # para el ajuste
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def diff(v):
    v_diff = []
    for i in range(1, len(v) - 1):
        v_diff.append(v[i][0] - v[i - 1][0])
    return v_diff


# funcion para calibrar la escala del calibre. Toma un vector y devuelve el promedio de las diferencias entre elementos de ese vector
# junto con su desviación estandar
def calibrateScale(x):
    v_diff = []
    v_diff = diff(x)
    mm = np.mean(v_diff)
    mm_err = 2 * np.std(v_diff)
    return mm, mm_err


# kernel en el codigo de Richi (de donde sale este kernel? Porque usando este kernel la convolucion es tan ruidosa?)
s = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
# %% CARGA DEL MODO GAUSSIANO

# path = 'Modo Tem00.jpeg'
path = "Tem00-exposicion 280.png"
# cargamos la imagen a analizar
# AA = imageio.imread(path)
A = cv2.imread(path, 0)
A_copy = A.copy()
# layer de la imagen como matriz con valores de 0-255
# A = AA[:,:,1]# tomo un canal que no sature

# grafica la imagen original


linea_00 = 176
image = cv2.line(A_copy, (0, linea_00),
                 (np.shape(A)[1], linea_00), (255, 0, 0), 2)
# image = cv2. resize(image, (600, 800)) # Resize image.
window_name = "linea"
cv2.imshow(window_name, image)
A2 = A.copy()[118:233, 232:377]
# %%
# Dibujar líneas de contorno
# Dibujar cara
h, w = A2.shape

c = np.meshgrid(np.arange(0, w), np.arange(0, h))
fig = plt.figure(figsize=(10, 10))
plt.rcParams['axes.facecolor'] = '#f6f6f6'
fig.set_facecolor('#f6f6f6')
ax1 = Axes3D(fig)

# ax1.w_xaxis.set_pane_color((0, 0, 0, 1))
# ax1.w_yaxis.set_pane_color((0, 0, 0, 1))
# ax1.zaxis.set_pane_color((0, 0, 0, 1))

# Este color es mejor
surf = ax1.plot_surface(c[0], c[1], A2, cmap="gist_heat")
fig.colorbar(surf, shrink=0.3, aspect=12)
ax1.set_title("Modo Tem01 ")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Intensidad [bits]")

ax1.contour(c[0], c[1], A2, zdir='x', offset=0)

plt.show()

# %% TOMA DEL PERFIL Y AJUSTE

# El perfil se tomo a ojo (al menos para el modo gaussiano)
perf = A[linea_00, :]

plt.plot(range(len(perf)), perf)


# %%
y = perf

x = range(len(perf))

parametros_optimos, matriz_de_cov = curve_fit(
    gauss, x, y, sigma=np.sqrt(y) + 1, p0=[500, 629, 100]
)
error = np.sqrt(np.diag(matriz_de_cov))

# plt.errorbar(x, y, yerr=np.sqrt(y), fmt="+", label="Datos")
# plt.plot(x, gauss(x, *parametros_optimos), ".", label="Ajuste")


plt.figure(1, figsize=(8, 4))
plt.clf()
plt.rc("font", family="serif", size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
plt.rcParams['axes.facecolor'] = '#ffffff'
fig.set_facecolor('#ffffff')
# 1 Muestro la referencia.
ax.errorbar(x, y, fmt="+", label="Mediciones")
ax.plot(x, gauss(x, *parametros_optimos), "-", label="Ajuste")
ax.set_xlabel("Pixeles [px]")
ax.set_ylabel("Intensidad [bits]")
ax.ticklabel_format(axis="y", style="sci", scilimits=(1, 4))
ax.set_title("Ajuste TEM00")
ax.legend(loc="upper left")
ax.set_xlim(150, 450)
ax.grid(True)
# ax.set_xlim(prelim[0].min(), prelim[0].max())
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.subplots_adjust(bottom=0.137, left=0.09, right=0.976, top=0.923)
plt.show()
# plt.savefig('..\Curva P vs I.png') # windows
plt.savefig("../Ajuste perfil gaussiano .png")  # linux

# plt.yscale('log')
print(f" Parametros = {parametros_optimos}\n errores={error}")
# plt.title("Ajuste Gaussiano")
# Analisis del estadistico χ²
# esto es: grados de libertad = #mediciones - #Parámetros - 1
grados_lib = len(y) - len(parametros_optimos) - 1
modelo = gauss(x, *parametros_optimos)
error_y = np.sqrt(y)
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
# %% CARGA DE LA SEGUNDA IMAGEN A ANALIZAR <<MODO TEM01>>

# path = 'Modo Tem00.jpeg'
path = "Tem01-exposicion770.png"
# cargamos la imagen a analizar
# AA = imageio.imread(path)
A = cv2.imread(path, 0)
print(A.shape)
A_copy = A.copy()
# layer de la imagen como matriz con valores de 0-255
# A = AA[:,:,1]# tomo un canal que no sature

# grafica la imagen original
linea_01 = 332  # x que utilizo para tomar el perfil

image = cv2.line(A_copy, (linea_01, 0),
                 (linea_01, np.shape(A)[0]), (255, 0, 0), 2)
# image = cv2. resize(image, (600, 800)) # Resize image.
window_name = "linea"
cv2.imshow(window_name, image)
A2 = A.copy()[150:340, 220:440]
# %% MODO TEM01 EN 3D

# Dibujar líneas de contorno
# Dibujar cara
h, w = A2.shape

c = np.meshgrid(np.arange(0, w), np.arange(0, h))
fig = plt.figure(figsize=(10, 10))
plt.rcParams['axes.facecolor'] = '#f6f6f6'
fig.set_facecolor('#f6f6f6')
ax1 = Axes3D(fig)
# ax1.w_xaxis.set_pane_color((0, 0, 0, 1))
# ax1.w_yaxis.set_pane_color((0, 0, 0, 1))
# ax1.zaxis.set_pane_color((0, 0, 0, 1))

# Este color es mejor
surf = ax1.plot_surface(c[0], c[1], A2, cmap="gist_heat")
fig.colorbar(surf, shrink=0.3, aspect=12)
ax1.set_title("Modo Tem01 ")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel(" Intensidad [bits]")

ax1.contour(c[0], c[1], A2, zdir='x', offset=0)

plt.show()

# %% TOMA DEL PERFIL Y AJUSTE

# El perfil se tomo a ojo (al menos para el modo gaussiano)
perf = A[:, linea_01]

plt.plot(np.arange(len(perf)), perf)


# %% AJUSTE MODO TEM 01
y = np.array(perf)

x = np.arange(len(perf))

parametros_optimos, matriz_de_cov = curve_fit(TEMX01, x, y, p0=[250, 150, 100])
error = np.sqrt(np.diag(matriz_de_cov))

# plt.errorbar(x, y, yerr=np.sqrt(y), fmt="+", label="Datos")
# plt.plot(x, TEMX01(x, *parametros_optimos), ".", label="Ajuste")

plt.figure(1, figsize=(8, 4))
plt.clf()
plt.rc("font", family="serif", size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
plt.rcParams['axes.facecolor'] = '#ffffff'
fig.set_facecolor('#ffffff')
# 1 Muestro la referencia.
ax.errorbar(x, y, fmt="+", label="Mediciones")
ax.plot(x, TEMX01(x, *parametros_optimos), "-", label="Ajuste")
ax.set_xlabel("Pixeles [px]")
ax.set_ylabel("Intensidad [bits]")
ax.ticklabel_format(axis="y", style="sci", scilimits=(1, 4))
ax.set_title("Ajuste Tem01")
ax.legend(loc="upper left")
ax.grid(True)
ax.set_xlim(140, 350)
# ax.set_xlim(prelim[0].min(), prelim[0].max())
# Ordenamos y salvamos la figura.
# plt.tight_layout()
plt.subplots_adjust(bottom=0.137, left=0.090, right=0.976, top=0.923)

plt.show()
# plt.savefig('..\Curva P vs I.png') # windows
plt.savefig("../Ajuste perfil tem01.png")  # linux

# plt.yscale('log')
print(f" Parametros = {parametros_optimos}\n errores={error}")
# Analisis del estadistico χ²
# esto es: grados de libertad = #mediciones - #Parámetros - 1
grados_lib = len(y) - len(parametros_optimos) - 1
modelo = TEMX01(x, *parametros_optimos)
error_y = np.sqrt(y)
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
# %%AJUSTE USANDO MINIMIZE

# def ajuste2(x,p): return (p[0]*x)/(np.log(x)+p[1]) # modelo


def ajuste2(x, x0, I, w):
    Hp = special.hermite(1, monic=True)
    U = Hp((np.sqrt(2) * np.array(x - x0)) / w) * np.exp(
        -(np.array(x - x0) ** 2) / w ** 2
    )
    return I * U ** 2


x1 = np.array(x) + 1
y1 = np.array(y) + 1
# y1[2]=355

# Errores
error_pd4 = np.array(x) * 0.01
error_v4 = np.array(y) * 0.1

p = parametros_optimos
pp0 = np.concatenate((p, x1))


def chi_2(pp):
    xfit = pp[3: len(pp)]
    p = pp[0:3]
    chi = sum(
        ((y1 - ajuste2(xfit, *p)) / error_v4) ** 2 +
        ((x1 - xfit) / error_pd4) ** 2
    )
    return chi


# print(chi_2(pp0))
minimizada = minimize(chi_2, pp0, method="BFGS")

# fig, ax = plt.subplots()
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot()
# ax.plot(x1,y1,".",label='Datos')
ax.errorbar(x1, y1, error_v4, error_pd4, fmt="+", label="Datos")
# plt.rcParams['axes.facecolor'] = '#f6f6f6'
# fig.set_facecolor('#f6f6f6')

ax.plot(
    np.linspace(x1.min(), x1.max(), 1000),
    ajuste2(np.linspace(x1.min(), x1.max(), 1000), *minimizada.x[0:3]),
    label="Ajuste usando minimize",
)
ax.plot(
    np.linspace(x1.min(), x1.max(), 1000),
    TEMX01(np.linspace(x1.min(), x1.max(), 1000), *parametros_optimos),
    label="Ajuste usando curve_fit",
)
# plt.plot( np.linspace(x.min(),x.max(),100),f(np.linspace(x.min(),x.max(),100),*params))

ax.set_title("Ajuste perfil de modo TEM01", fontsize=18)
ax.ticklabel_format(axis="x", style="sci", scilimits=(1, 4))

ax.set_xlabel("pixel [px]", fontsize=15)
ax.set_ylabel("Intensidad [bits]", fontsize=15)
# ax.xticks(fontsize=13)
# ax.yticks(fontsize=13)
# ax.legend(fontsize=15)
# ax.grid(True)
# ax.set_xscale('log')
# plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.subplots_adjust(bottom=0.114, left=0.124, right=0.996, top=0.936)
plt.show()
plt.savefig("../ajuste curv y mini Tem01.png")
# %% CARGA DE LA SEGUNDA IMAGEN A ANALIZAR <<MODO TEM03>>

# path = 'Modo Tem00.jpeg'
path = "Tem30-exposicion 230.png"
# cargamos la imagen a analizar
# AA = imageio.imread(path)
A = cv2.imread(path, 0)

# roto la imagen apra analizarla con el codigo anterior
rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1)
A = cv2.warpAffine(A, rotationMatrix, (w, h))
A_copy = A.copy()
# grafica la imagen original
linea_01 = 353  # x que utilizo para tomar el perfil

image = cv2.line(A_copy, (linea_01, 0),
                 (linea_01, np.shape(A)[0]), (255, 0, 0), 2)
# image = cv2. resize(image, (600, 800)) # Resize image.
window_name = "linea"
h, w = A.shape


cv2.imshow(window_name, image)

# %% MODO TEM01 EN 3D

# Dibujar líneas de contorno
# Dibujar cara


h, w = A.shape

c = np.meshgrid(np.arange(0, w), np.arange(0, h))
fig = plt.figure(figsize=(8, 6))
ax1 = Axes3D(fig)
surf = ax1.plot_surface(c[0], c[1], A, cmap="bone")  # Este color es mejor
fig.colorbar(surf, shrink=0.3, aspect=12)
ax1.set_title("Modo Tem01 ")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Intensidad [bits]")

ax1.contour(c[0], c[1], A, zdir="x", offset=0)

plt.show()

# %% TOMA DEL PERFIL Y AJUSTE

# El perfil se tomo a ojo (al menos para el modo gaussiano)
perf = A[:, linea_01]

plt.plot(np.arange(len(perf)), perf)

# %% AJUSTE MODO TEM 03


def TEMX03(x, x0, I, w):
    Hp = special.hermite(3, monic=False)
    U = Hp((np.sqrt(2) * np.array(x - x0)) / w) * np.exp(
        -(np.array(x - x0) ** 2) / w ** 2
    )
    return I * U ** 2


y = np.array(perf)

x = np.arange(len(perf))

parametros_optimos, matriz_de_cov = curve_fit(TEMX03, x, y, p0=[280, 5, 50])
error = np.sqrt(np.diag(matriz_de_cov))

plt.errorbar(x, y, yerr=np.sqrt(y), fmt="+", label="Datos")
plt.plot(x, TEMX03(x, *parametros_optimos), ".", label="Ajuste")
# plt.yscale('log')
print(f" Parametros = {parametros_optimos}\n errores={error}")
plt.title("Ajuste Tem01")
# Analisis del estadistico χ²
# esto es: grados de libertad = #mediciones - #Parámetros - 1
grados_lib = len(y) - len(parametros_optimos) - 1
modelo = TEMX03(x, *parametros_optimos)
error_y = np.sqrt(y)
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
# %%AJUSTE USANDO MINIMIZE


# def ajuste2(x,p): return (p[0]*x)/(np.log(x)+p[1]) # modelo

def ajuste2(x, x0, I, w):
    Hp = special.hermite(3, monic=True)
    U = Hp((np.sqrt(2) * np.array(x - x0)) / w) * np.exp(
        -(np.array(x - x0) ** 2) / w ** 2
    )
    return I * U ** 2


x1 = np.array(x) + 1
y1 = np.array(y) + 1
# y1[2]=355

# Errores
error_pd4 = np.array(x) * 0.01
error_v4 = np.array(y) * 0.1

p = parametros_optimos
pp0 = np.concatenate((p, x1))


def chi_2(pp):
    xfit = pp[3: len(pp)]
    p = pp[0:3]
    chi = sum(
        ((y1 - ajuste2(xfit, *p)) / error_v4) ** 2 +
        ((x1 - xfit) / error_pd4) ** 2
    )
    return chi


# print(chi_2(pp0))
minimizada = minimize(chi_2, pp0, method="BFGS")

# fig, ax = plt.subplots()
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot()
# ax.plot(x1,y1,".",label='Datos')
ax.errorbar(x1, y1, error_v4, error_pd4, fmt="+", label="Datos")
# plt.rcParams['axes.facecolor'] = '#f6f6f6'
# fig.set_facecolor('#f6f6f6')

ax.plot(
    np.linspace(x1.min(), x1.max(), 1000),
    ajuste2(np.linspace(x1.min(), x1.max(), 1000), *minimizada.x[0:3]),
    label="Ajuste usando minimize",
)
ax.plot(
    np.linspace(x1.min(), x1.max(), 1000),
    TEMX03(np.linspace(x1.min(), x1.max(), 1000), *parametros_optimos),
    label="Ajuste usando curve_fit",
)
# plt.plot( np.linspace(x.min(),x.max(),100),f(np.linspace(x.min(),x.max(),100),*params))

ax.set_title("Ajuste perfil de modo TEM01", fontsize=18)
ax.ticklabel_format(axis="x", style="sci", scilimits=(1, 4))

ax.set_xlabel("pixel [px]", fontsize=15)
ax.set_ylabel("Intensidad [bits]", fontsize=15)
# ax.xticks(fontsize=13)
# ax.yticks(fontsize=13)
# ax.legend(fontsize=15)
# ax.grid(True)
# ax.set_xscale('log')
# plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.subplots_adjust(bottom=0.114, left=0.124, right=0.996, top=0.936)
plt.show()
plt.savefig("../ajuste curv y mini Tem01.png")

# %% PLOTEO DE CURVA POTENCIA DE LASER ROJO VS POTENCIA DE LASER VERDE
# windows
os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Laser\Mediciones')

# os.chdir(
#     "/media/daniel/OS/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Laser/Mediciones"
# )  # linux

data_inf = np.loadtxt(r"P vs I luz infrarroja (prisma).csv",
                      delimiter=";", skiprows=1)[:25]

data_verd = np.loadtxt(r"P vs I luz verde.csv",
                       delimiter=";", skiprows=1)[35:60]
# %% CURVA DE  POTENCIA VERDE VS POTENCIA INFRAROJA
I_inf = data_inf[:, 0]
P_inf = data_inf[:, 1]

I_verd = data_verd[:, 0]
P_verd = data_verd[:, 1]/1000  # lo paso a mW

error_y = data_inf[:, 2]/1000

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.rc("font", family="serif", size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
plt.rcParams['axes.facecolor'] = '#ffffff'
fig.set_facecolor('#ffffff')
# 1 Muestro la referencia.
ax.errorbar(P_inf, P_verd, error_y, fmt="+", label="Mediciones")
ax.set_xlabel("Potencia Laser infrarrojo [$mw$]")
ax.set_ylabel("Potencia Laser verde [$mw$]")
ax.ticklabel_format(axis="y", style="sci", scilimits=(1, 4))
ax.set_title("Curva P vs I")
ax.legend(loc="lower right")
ax.grid(True)

# ax.set_xlim(prelim[0].min(), prelim[0].max())
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()
# plt.savefig('..\Curva P vs I.png') # windows
plt.savefig("../Curva P_verde vs P_rojo.png")  # linux

# %% CURVA DE POTENCIA VS CORRIENTE PARA LASER VERDE E INFRARROJO

plt.figure(1, figsize=(8, 4))
plt.clf()
plt.rc("font", family="serif", size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
plt.rcParams['axes.facecolor'] = '#ffffff'
fig.set_facecolor('#ffffff')
# 1 Muestro la referencia.
ax.errorbar(I_verd, P_verd/1e3, error_y/1000, fmt="+", label="Laser Verde")
ax.errorbar(I_inf, P_inf, error_y, fmt="+", label="Laser Infrarrojo")
ax.set_xlabel("Corriente [$A$]")
ax.set_ylabel("Potencia [$mw$]")
ax.ticklabel_format(axis="y", style="sci", scilimits=(1, 4))
ax.set_title("Curva P vs I")
ax.legend(loc="lower left")
ax.set_yscale('log')
ax.grid(True)

# ax.set_xlim(prelim[0].min(), prelim[0].max())
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.subplots_adjust(bottom=0.137, left=0.105, right=0.976, top=0.923)
plt.show()
# plt.savefig('..\Curva P vs I.png') # windows
plt.savefig("../Curva P vs I para ambos laser.png")  # linux
# %% CURVA DE POTENCIA VS CORRIENTE PARA LASER INFRARROJO

plt.figure(1, figsize=(8, 4))
plt.clf()
plt.rc("font", family="serif", size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
plt.rcParams['axes.facecolor'] = '#ffffff'
fig.set_facecolor('#ffffff')
# 1 Muestro la referencia.
# ax.errorbar(I_verd, P_verd/1e3, error_y/1000, fmt="+", label="Laser Verde")
ax.errorbar(I_inf, P_inf, error_y, fmt="+", label="Laser Infrarrojo")
ax.set_xlabel("Corriente [$A$]")
ax.set_ylabel("Potencia [$mw$]")
ax.ticklabel_format(axis="y", style="sci", scilimits=(1, 4))
ax.set_title("Curva P vs I")
ax.legend(loc="upper left")
# ax.set_yscale('log')
ax.grid(True)

# ax.set_xlim(prelim[0].min(), prelim[0].max())
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.subplots_adjust(bottom=0.137, left=0.07, right=0.976, top=0.923)
plt.show()
# plt.savefig('..\Curva P vs I.png') # windows
plt.savefig("../Curva P vs I para laser infrarrojo.png")  # linux
