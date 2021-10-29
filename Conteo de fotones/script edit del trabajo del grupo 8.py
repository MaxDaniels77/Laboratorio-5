# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:52:47 2021

@author: cyber
"""

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
os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Conteo de fotones\Dia 3\Grupo v3')

# %% Funciones


def la_ajustadora(funcion, x_data, y_data, y_error, parametros_iniciales, nombres_de_los_parametros):
    parametros_optimos, matriz_de_cov = curve_fit(
        funcion, x_data, y_data, sigma=y_error, p0=parametros_iniciales)
    desvios_estandar = np.sqrt(np.diag(matriz_de_cov))
    datos = pd.DataFrame(index=nombres_de_los_parametros)
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


def BE(n, n_, a):
    return (a*n_**n)/((1 + n_)**(1 + n))


def gauss0(x, a, c):
    return a*np.exp(-(x)**2/(2*c**2))


def gauss(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))


def la_real(x, x1, A, B):
    return A*poisson.pmf(x, x1) + (B*x1**x)/(1 + x1)**(1 + x)


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
        if len(Intensidades) - delta >= pixel >= delta:
            entorno = Intensidades[pixel-delta:pixel+delta]
            if intensidad >= max(entorno) >= umbral and intensidad != Intensidades[pixel-1]:
                ubi_picos.append(pixel)
    return np.array(ubi_picos)


# %% ARCHIVO QUE SE USO PARA DETERMINAR LA CANTIDAD DE PUNTOS POR PICO (SE FIJO ARBITRARIAMENTE EN 5)
prelim = np.loadtxt(
    r'D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo 3v/data_preliminar_100ohm 25ns.txt', delimiter=',')

plt.figure(1, figsize=(10, 6))
plt.clf()
plt.rc('font', family='serif', size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
# 1 Muestro la referencia.
ax.plot(prelim[0], prelim[1], label='Osc. screen')
ax.set_xlabel("Tiempo[s]")
ax.set_ylabel("Voltaje[V]")
ax.ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
ax.set_title('Pulso 25 ns')
ax.legend(loc='lower right')
ax.grid(True)
ax.set_xlim(prelim[0].min(), prelim[0].max())
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()
plt.savefig('pulso 25 ns.png')

# %% ESTIMACION DEL TIEMPO DE CORRELACION

DatosC = np.loadtxt(
    'Pantalla para correlacion 25 ms 500 mv_medicion2.txt', delimiter=',')
DatosC = np.loadtxt(
    'Pantalla para correlacion 25 ms 500 mv.txt', delimiter=',')
t = DatosC[0]
d = DatosC[1]
''' # Ultima medicion que hizo lean para determinar la correlacion
DatosC = np.loadtxt('pantalla autocorrelacion 250ms (ultima que hice)', delimiter = ',')
t = DatosC[0][150:2300] 
d = DatosC[1][150:2300]
'''
# %% PLOTEO DE VERIFICACION

plt.plot(t, d)
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
plt.title("Pantalla")

# %% PLOTEO DE LA CORRELACION

prom = np.mean(d)
# prom = 0
Corre = np.correlate(d - prom, d - prom, mode="full")
print(len(Corre))

paso = t[1]-t[0]
ti_c = np.linspace(-t[-1], t[-1], len(t)*2-1)
plt.plot(ti_c, Corre)
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Correlacion')
plt.title('Autocorrelacion')

# %% DETERMINACION DEL TIEMPO DE CORRELACION
'''FALTA CORREGIR'''
# Busca Tc, busca el ancho a mitad de altura


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


corr_calc = busca_Tc(paso, Corre)
print(f'La correlacion es: {corr_calc}')


# %% CARGA DE DATOS PARA DETERMINAR NUMERO DE PICOS


os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Conteo de fotones\Dia 3\Grupo v3')

Pantalla = np.load(r'1000 pantallas 500 ns 10mv (2).npz')

for item in Pantalla.keys():
    print(item)

for item in Pantalla.values():
    print(item)

np.savetxt('datos_analisis ultiom0.csv', np.array(
    Pantalla['screens']), delimiter=',')

# %% PRINT DE DATOS CRUDOS

Pantalla = Pantalla['screens']
plt.plot(range(len(Pantalla)), Pantalla)
plt.title("Pulso")
plt.ylabel("Voltaje[V]")
plt.xlabel("Tiempo[s]")
# plt.xlim([0,0.010])
plt.grid(True)

# %% HISTOGRAMA DE LOS DATOS CRUDOS
plt.figure(1, figsize=(8, 6))
plt.clf()
plt.rc('font', family='serif', size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
# 1 Muestro la referencia.
a = ax.hist(Pantalla, bins=200, label='Histograma')
ax.set_xlabel("Alturas [mV]")
ax.set_ylabel("Apariciones")
ax.ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
ax.set_title('Histograma de datos crudos')
ax.set_yscale('log')
ax.legend()
ax.grid(True)
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()
plt.savefig('Histograma de los datos crudos')
# %% CONFIGURACION DE ESCALAS DEMASES

data = Pantalla
escala = 1  # microsegundos
resolucion = escala/250*1e-6  # Consutla
tiempo = np.arange(0, resolucion*(len(data)), resolucion)

# %% BUSQUEDA DE PICOS

alturas = []
t_a = []
delta = 5  # numero de puntos que fijamos por pico

ubi_picos1 = busca_picos(-data, delta, 0.0000001)
ubi_picos2 = busca_picos(data, delta, 0.0000001)

for index in ubi_picos1:
    alturas.append(data[index])
    t_a.append(tiempo[index])

for index in ubi_picos2:
    alturas.append(data[index])
    t_a.append(tiempo[index])

alturas = np.array(alturas)
print(len(alturas))

# %% VERIFICACION DE LA DETECCION DE PICOS

plt.figure()
plt.plot(tiempo, data)
# plt.plot(tiempo[0:len(tiempo)]*1e6, data[0:len(tiempo)]*1e3)
plt.plot(np.array(t_a[0:len(t_a)]), np.array(
    alturas[0:len(t_a)]), '.', color='red')
plt.title("Deteccion de picos")
plt.ylabel("Voltaje[mV]")
plt.xlabel("Tiempo[us]")
plt.grid(True)

# %% HISTOGRAMA DE LOS PICOS
# %%
### Histograma ###
alturas = np.array(alturas[0:len(t_a)])  # consultar esto
index_peaks_restric = np.where(alturas > -0.02)

plt.figure(figsize=(10, 6))
cantidad_de_bins = (2 + 3*np.log2(len(alturas)))
lista_bins = np.linspace(-0.02,
                         alturas[index_peaks_restric].max(), int(cantidad_de_bins))
hist, bins = np.histogram(
    alturas[index_peaks_restric], bins=lista_bins, density=False)
bins = bins[:-1]
###
plt.bar(bins, hist, width=0.0003, label='Datos')
plt.yscale('log')
# Configuración
plt.grid()
### Ejes ###
plt.axis('tight')
# plt.xlim((-1, 10))
# plt.ylim((0, 0.25))
plt.xlabel('Numero de eventos', fontsize=22)
plt.ylabel('Frecuencias relativas', fontsize=22)
plt.tick_params(labelsize=20)

###
plt.legend(loc=0, fontsize=20)
plt.savefig('histograma.png', bbox_inches='tight')
# %%
# %% VERSION 2.0 DEL GRAFICO PARA QUE SE PUEDA VER DONDE CORTA
# EL UMBRAL CON LOS PICOS GENERADOS POR LOS FOTONES

alturas = np.array(alturas[0:len(t_a)])  # consultar esto
index_peaks_restric = np.where(alturas > -0.02)

plt.figure(figsize=(10, 6))
cantidad_de_bins = (2 + 3*np.log2(len(alturas)))

lista_bins = np.linspace(-0.02,
                         alturas.max(), int(cantidad_de_bins))
hist, bins = np.histogram(
    alturas[index_peaks_restric], bins=lista_bins, density=False)
bins = bins[:-1]

# Generamos una figura
plt.figure(1, figsize=(10, 6))
plt.clf()
plt.rc('font', family='serif', size=13)
fig, ax = plt.subplots(2, 1, num=1, sharex=True)
# PICOS CON EL UMBRAL

ax[0].plot(np.array(alturas[0:len(t_a)])[1250:2500], np.array(t_a[0:len(t_a)])[1250:2500], ".",
           label='Picos')
ax[0].plot(np.array(alturas[0:len(t_a)])[1250:2500],
           np.array(t_a[0:len(t_a)])[1250:2500], "-")

# ax[0].plot(data[0:len(t_a)][1250:2500],tiempo[0:len(t_a)][1250:2500],linewidth=0.5,label='Recorte de medición')

ax[0].vlines(-0.003, np.array(t_a[0:len(t_a)])[1250:2500].min(),
             np.array(t_a[0:len(t_a)])[1250:2500].max(), ls="--", label='Threshold', color='green')
ax[0].ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
ax[0].set_xlim(-0.021324655172413793, 0)
ax[0].set_ylabel('Tiempo [$\mu s$]', fontsize=13)
ax[0].grid(True)
ax[0].legend()
ax[0].set_title('Histograma de picos', fontsize=16)

# 2HISTOGRAMA

ax[1].bar(bins, hist, width=0.0003, label='Datos')
ax[1].set_ylabel('Apariciones', fontsize=13), ax[1].grid(True)
ax[1].set_yscale('log')

# ax[1].set_ylim(100)

ax[1].set_xlabel('Alturas[mV]', fontsize=13)
ax[1].vlines(-0.003, 100, 1000000, ls="--",
             label='Threshold', color='green'), ax[1].legend()

# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()
plt.savefig('histograma edit propio.png')

# %%
'''Para que el histograma se vea mas agradable a la vista, voy a 
sacar del grafico los puntos que son menores auqe -0.07v aunque los
voy a seguir considerando para el analisis'''
alturas = np.array(alturas[0:len(t_a)])  # consultar esto

index_peaks_restric = np.where(alturas > -0.02)
plt.figure(1, figsize=(8, 6))
plt.clf()
plt.rc('font', family='serif', size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
# 1 Muestro la referencia.
values = ax.hist(alturas[index_peaks_restric], bins=125, label='Histograma')
# sns.histplot(x=alturas)
ax.set_xlabel("Alturas [mV]")
ax.set_ylabel("Apariciones")
ax.ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
ax.set_title('Histograma de los picos')
ax.set_yscale('log')
ax.legend()
ax.grid(True)
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()
plt.savefig('Histograma de los picos.png')

# %% ALTURAS RECORTANDO LOS PICOS

'''Aca añado una parte en particular para considerar el error, voy a calcular 
el histograma de el threshold + el error instrumental y - el error instrumental
y voy a tomar como error la diferencia entre ambos'''
threshold = -0.003
res_volt = 10e-3/250  # en volts
t = np.array(t_a[0:len(t_a)])
peak = np.array(alturas[0:len(t_a)])
peak_cut_index = np.where(peak < threshold)
peak_cut_index_sup = np.where(peak < threshold + res_volt)
peak_cut_index_inf = np.where(peak < threshold - res_volt)
plt.plot(t[peak_cut_index], peak[peak_cut_index])

# %% HISTOGRAMA DE PICOS RECORTADOS

final_peaks = peak[peak_cut_index]
index_final_peaks_restric = np.where(final_peaks > -0.02)

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.rc('font', family='serif', size=13)
fig, ax = plt.subplots(1, 1, num=1, sharex=True)
# 1 Muestro la referencia.- 75 BINS PARA EL HISTOGRAMA DE 1 US
a = ax.hist(final_peaks[index_final_peaks_restric],
            bins=83, label='Histograma')
ax.set_xlabel("Alturas [mV]")
ax.set_ylabel("Apariciones")
ax.ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
ax.set_title('Histograma de datos crudos')
ax.set_yscale('log')
ax.legend()
ax.grid(True)
# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()
plt.savefig('Histograma de cuentas apagadas.png')
# %% CONTEO DE PICOS POR PANTALLA

"""
Aca arranca la estadistica de Poisson o BE,
la idea es que busca los picos que hay en cada pantalla (2500)
en cada una de las Tt pantallas, que superren el umbral. Despues divide
la pantalla en fraccion partes y asigna a cada una la cantidad de picos 
que se detectaron ahi, eso es N1_0, que finalmente se concatena con N1 que 
tiene las apariciones de picos en cada ventana. 
"""

N1 = []

fraccion = 7
umbral1 = 0.003  # medio a ojo
distancia = 5
frac_de_ventanas = 1
puntos_por_ventana = int(2500/frac_de_ventanas)
cant_de_mediciones = 1001  # numero de mediciones + 1
Tt = cant_de_mediciones*frac_de_ventanas

for i in range(Tt):
    # print(1000-i)
    d = -data[i*puntos_por_ventana: (i+1)*puntos_por_ventana]
    picos1 = busca_picos(d, distancia, umbral1)
    N1_0 = np.zeros(int(fraccion))
    for i in range(int(fraccion)):
        for pic in picos1:
            if i*(puntos_por_ventana//fraccion) - 1 < pic <= (i + 1)*(puntos_por_ventana//fraccion) - 1 <= puntos_por_ventana:
                N1_0[i] += 1
    N1 = N1 + N1_0.tolist()
    # print(len(picos1))

# hISTOGRAMA DE PICOS POR PANTALLA

bins = np.arange(-0.5, 15.5, 1)

plt.figure()
bose_data = plt.hist(N1, bins, color='#0504aa', alpha=0.7,
                     rwidth=0.85, label='Mediciones')
# plt.plot(Torres, P_plot2*Mod, 'o', markersize = 10, color = 'red', label = 'Ajuste Poissoniano')
# plt.plot(Torres, P_plot1*Mod, 'o', markersize = 10, color = 'black', label = 'Ajuste BE')
plt.grid(True)
plt.legend()
plt.xlabel('Número de fotocuentas')
plt.ylabel('Apariciones')
plt.title('Histograma - ventana: 1us')

# %% AJUSTE DE BE
y_index = np.where(bose_data[0] != 0)
y = np.array(bose_data[0][y_index])[:6]

x = np.array(range(len(bose_data[0][y_index])))[:6]

parametros_optimos, matriz_de_cov = curve_fit(
    BE, x, y, sigma=np.sqrt(y), p0=[1, 3000])
error = np.sqrt(np.diag(matriz_de_cov))
X = np.linspace(x.min(), x.max(), 1000)


plt.errorbar(x, y, yerr=np.sqrt(y), fmt='+', label='Datos')
plt.plot(x, BE(x, *parametros_optimos), ".", label='Ajuste')
plt.yscale('log')
print(f' Parametros = {parametros_optimos}\n errores={error}')

# %%

# Analisis del estadistico χ²
# esto es: grados de libertad = #mediciones - #Parámetros - 1
grados_lib = len(y) - len(parametros_optimos)-1
modelo = BE(x, *parametros_optimos)
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

# %%

y_index = np.where(bose_data[0] != 0)
y = np.array(bose_data[0][y_index])[:4]

x = np.array(range(len(bose_data[0][y_index])))[:4]

parametros_optimos, matriz_de_cov = curve_fit(
    poiss1, x, y, p0=[1.5, 4000])
error = np.sqrt(np.diag(matriz_de_cov))

plt.errorbar(x, y, yerr=np.sqrt(y), fmt='+', label='Datos')
plt.plot(x, poiss1(x, *parametros_optimos), ".", label='Ajuste')
plt.yscale('log')
plt.legend()
print(f' Parametros = {parametros_optimos}\n errores={error}')

# %% ANALISIS DIST DE POISSON

# Analisis del estadistico χ²
# esto es: grados de libertad = #mediciones - #Parámetros - 1
# necesito eliminar las columnas que son cero, ya que el error depende de la altura
# del bin
y_index = np.where(y != 0)
y = y[y_index]
x = x[y_index]

grados_lib = len(y) - len(parametros_optimos) - 1
modelo = poiss1(x, *parametros_optimos)
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


# %%
mediciones = np.array(alturas)
# mediciones = data*1e3
DV = 0.08

bins = np.arange(-14, 4, DV) - DV/2

plt.figure(figsize=(10, 6))
plt.hist(mediciones, bins, color='#0504aa',
         alpha=0.7, rwidth=0.85, label='Mediciones')
# plt.plot(r_plot, P_plot, label = 'Ajuste Gaussiano', color = 'red')
plt.grid(True)
plt.legend()
plt.xlabel('Alturas[mV]')
plt.ylabel('Apariciones')
plt.title('Histograma de picos')
plt.yscale("log")
# plt.ylim([100, 100000])
# %% VERSION 2.0 DEL GRAFICO PARA QUE SE PUEDA VER DONDE CORTA
# EL UMBRAL CON LOS PICOS GENERADOS POR LOS FOTONES

# Generamos una figura
plt.figure(1, figsize=(10, 6))
plt.clf()
plt.rc('font', family='serif', size=13)
fig, ax = plt.subplots(2, 1, num=1, sharex=True)
# 1 Muestro la referencia.
ax[0].plot(data[1500:2500]*1000, tiempo[1500:2500],
           linewidth=0.5, label='Recorte de medición')
ax[0].vlines(-2.5, tiempo[1500:2500].min(),
             tiempo[1500:2500].max(), ls="--", label='Threshold')
#ax[0].plot(t, Referencia_y, label='Referencia')
ax[0].set_ylabel('Tiempo [$\mu s$]', fontsize=13)
ax[0].grid(True)
ax[0].legend()
ax[0].set_title('Histograma de datos', fontsize=16)
# 2 Muestro la señal
ax[1].hist(mediciones, bins, color='#0504aa',
           alpha=0.7, rwidth=0.85, label='Mediciones')
ax[1].set_ylabel('Apariciones', fontsize=13), ax[1].grid(True)
ax[1].set_yscale('log')
ax[1].set_ylim(100)
ax[1].set_xlabel('Alturas[mV]', fontsize=13)
ax[1].vlines(-2.5, 100, 1000000, ls="--", label='Threshold'), ax[1].legend()

# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()
plt.savefig('histograma edit propio.png')
# %% AJUSTE DE GAUSSIANA DE CUENTAS OSCURAS

'''AL usar plt.hist como se configuro previamente es necesario matar los puntos que quedan
pegados en y = 0 y tomar los maximos correspondientes, para esto se realiza el siguiente 
crop en los datos'''

x = a[1][np.where(a[0] > 0)]
y = a[0][np.where(a[0] > 0)]
print(y.max())
# y = y/y.max() # normalice para que el algoritmo converja mejor

# plt.plot(x,y,".") # Chequeo que lo que obtuve sea correcto

parametros_optimos, matriz_de_cov = curve_fit(
    gauss, x, y, p0=[500000, 0.6, 0.8])
error = np.sqrt(np.diag(matriz_de_cov))
X = np.linspace(x.min(), x.max(), 1000)
plt.plot(x, y, ".")
plt.plot(X, gauss(X, *parametros_optimos))
print(f' Parametros = {parametros_optimos}\n errores={error}')
threshold = parametros_optimos[1]-4*parametros_optimos[2]
print(threshold)
# %%


# %% RECORTANDO EN EL UMBRAL FIJADO A OJO EN -2.5
#    recortando en el umbralllll

data_escalada = data*1e3
index_recorte = np.where(data_escalada < -2.5)
# plt.hist(data_escalada[index_recorte],bins=100)
data = data[index_recorte]
tiempo = tiempo[index_recorte]

# %%

alturas = []
t_a = []
delta = 2

ubi_picos1 = busca_picos(-data, delta, 0.00001)
ubi_picos2 = busca_picos(data, delta, 0.00001)

for index in ubi_picos1:
    alturas.append(data[index])
    t_a.append(tiempo[index])

for index in ubi_picos2:
    alturas.append(data[index])
    t_a.append(tiempo[index])

print(len(alturas))

# %%

plt.figure()
plt.plot(tiempo, data)
# plt.plot(tiempo[0:len(tiempo)]*1e6, data[0:len(tiempo)]*1e3)
plt.plot(np.array(t_a[0:len(t_a)]), np.array(
    alturas[0:len(t_a)]), 'o', color='red')
plt.title("Deteccion de picos")
plt.ylabel("Voltaje[mV]")
plt.xlabel("Tiempo[us]")
# plt.xlim([0, 4])
# plt.ylim([-10, 2])
plt.grid(True)

# %%

mediciones = np.array(alturas)*1e3
# mediciones = data*1e3
DV = 0.08

bins = np.arange(-14, 4, DV) - DV/2

plt.figure(figsize=(10, 6))
plt.hist(mediciones, bins, color='#0504aa',
         alpha=0.7, rwidth=0.85, label='Mediciones')
# plt.plot(r_plot, P_plot, label = 'Ajuste Gaussiano', color = 'red')
plt.grid(True)
plt.legend()
plt.xlabel('Alturas[mV]')
plt.ylabel('Apariciones')
plt.title('Histograma de picos')
plt.yscale("log")
plt.ylim([100, 100000])
# %% VERSION 2.0 DEL GRAFICO PARA QUE SE PUEDA VER DONDE CORTA
# EL UMBRAL CON LOS PICOS GENERADOS POR LOS FOTONES

# Generamos una figura
plt.figure(1, figsize=(10, 6))
plt.clf()
plt.rc('font', family='serif', size=13)
fig, ax = plt.subplots(2, 1, num=1, sharex=True)
# 1 Muestro la referencia.
ax[0].plot(data[1500:2500]*1000, tiempo[1500:2500],
           linewidth=0.5, label='Recorte de medición')
ax[0].vlines(-2.5, tiempo[1500:2500].min(),
             tiempo[1500:2500].max(), ls="--", label='Threshold')
#ax[0].plot(t, Referencia_y, label='Referencia')
ax[0].set_ylabel('Tiempo [$\mu s$]', fontsize=13)
ax[0].grid(True)
ax[0].legend()
ax[0].set_title('Histograma de datos', fontsize=16)
# 2 Muestro la señal
ax[1].hist(mediciones, bins, color='#0504aa',
           alpha=0.7, rwidth=0.85, label='Mediciones')
ax[1].set_ylabel('Apariciones', fontsize=13), ax[1].grid(True)
ax[1].set_yscale('log')
ax[1].set_ylim(100, 1e5)
ax[1].set_xlabel('Alturas[mV]', fontsize=13)
ax[1].vlines(-2.5, 100, 1000000, ls="--", label='Threshold'), ax[1].legend()

# Ordenamos y salvamos la figura.
plt.tight_layout()
plt.show()
plt.savefig('histograma edit propio.png')


'''Puedeb estar faltando cosas aca'''


# %% Estudio de la correlacion, para cuando gira el plato mediciones en la escala del tiempo de correlacion
# sin resistencia conectada para usarlo de integrador

os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Conteo de fotones\Grupo v3 dia 2 disco d\Grupo v3')
# 5s

# DatosC = np.loadtxt('D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo v3 dia 2 disco d/Grupo v3/1 ventana determinacion de Tc, 5v,5s.csv', delimiter = ',')
# t = DatosC[0][255:2350] # recorte para el set de datos 5s
# d = DatosC[1][255:2350]

# 250 ms

# DatosC = np.loadtxt('D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo v3 dia 2 disco d/Grupo v3/1 ventana determinacion de Tc, 5v,250ms.csv', delimiter = ',')
# t = DatosC[0][:2190] # recorte set de datos 250 ms
# d = DatosC[1][:2190]

# 500 ms

DatosC = np.loadtxt(
    'D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo v3 dia 2 disco d/Grupo v3/1 ventana determinacion de Tc, 5v,500ms.csv', delimiter=',')
t = DatosC[0][20:2250]  # recorte set de datos
d = DatosC[1][20:2250]


# DatosC = np.load(r'D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Grupo v3 dia 2 disco d/Grupo v3/1 ventana determinacion de Tc, 5v,500ms.npz')


# %%

plt.plot(t, d)
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
plt.title("Pantalla")

# %%

prom = np.mean(d)
# prom = 0
Corre = np.correlate(d - prom, d - prom, mode="full")
print(len(Corre))

# %%

paso = t[1]-t[0]
ti_c = np.linspace(0, paso*2*len(t)-2, 2*len(t)-1)
plt.plot(ti_c, Corre)
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Correlacion')
plt.title('Autocorrelacion')

# %% Busca Tc, busca el ancho a mitad de altura


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


# %%
corr_calc = busca_Tc(paso, Corre)
print(f'La correlacion es: {corr_calc}')


# %%

Tcs = []
Tt = 51
paso = 0.001/250

for i in range(Tt):
    d0 = d[i*2500: (i+1)*2500] - np.mean(d[i*2500: (i+1)*2500])
    Corre0 = np.correlate(d0[:2500], d0[:2500], mode="full")
    Tc0 = busca_Tc(paso, Corre0)
    Tcs.append(Tc0)

print('Tc = ({:4.2f} ± {:4.2f})s'.format(np.mean(Tcs)*1000, np.std(Tcs)*1000))


# %%
Torres00 = np.arange(-0.76, 1.32, DV)
Torres0 = np.zeros(len(Torres00))
for k, element in enumerate(Torres00):
    Torres0[k] = np.round(element, 5)

Cuentas0 = np.zeros(len(Torres0))

for lugar, voltaje in enumerate(Torres0):
    for element in mediciones:
        if voltaje - DV < element <= voltaje + DV:
            Cuentas0[lugar] += 1
# %% Para ver si cuenta bien

plt.figure()
plt.plot(Torres0, Cuentas0, '.')
# plt.yscale("log")
plt.grid(True)

# %%
parametros1 = la_ajustadora(gauss, Torres0, Cuentas0, None, [
                            1000, 0.1, 0.1], ["a", 'b', "c"])
a0 = parametros1["Valor"]["a"]
b0 = parametros1["Valor"]["b"]
c0 = parametros1["Valor"]["c"]

r_plot = np.linspace(-5.2, 5.2, 1000)
P_plot = gauss(r_plot, a0, b0, c0)

plt.figure()
plt.plot(Torres0, Cuentas0, 'o', label='Mediciones')
plt.plot(r_plot, P_plot, label='Ajuste Gaussiano')
plt.grid(True)
plt.legend()
plt.xlabel('Voltaje [mV]')
plt.ylabel('Cantidad')
plt.title('Ruido electrico')
# plt.yscale('log')
plt.ylim([10, 700000])
# %%


# os.chdir('D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Conteo de fotones/Dia 3/Grupo v3/Nuevo analisis')

# mediciones = []
# for i in os.listdir("."):
#     if i.find('med') != -1:
#         mediciones.append(i)
#     if len(mediciones) == nMed:
#         break
# # Listar mediciones

# data0 = np.loadtxt(mediciones[0], delimiter=',')
# %%

os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Conteo de fotones\Dia 3\Grupo v3')

Pantalla = np.load(r'1000 pantallas 500 ns 10mv (2).npz')

for item in Pantalla.keys():
    print(item)

for item in Pantalla.values():
    print(item)
cuentas = []
thres = -5e-3
data0 = Pantalla['screens']
frac_de_ventanas = 2
puntos_por_ventana = int(2500/frac_de_ventanas)
nMed = 1000*frac_de_ventanas
for i in range(nMed):
    data = data0[puntos_por_ventana*i:puntos_por_ventana*(i+1)]
    minimos = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # Mínimos
    cuentas.append(data[np.where(data[minimos] < thres)].shape[0])
    # plt.plot(range(len(data)),data) # verificacion para una ventana
    # plt.plot(np.array(range(len(data)))[minimos],data[minimos],"o")


# data = data0
# minimos = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # Mínimos
# cuentas = data[minimos]

# plt.show()
# print(cuentas)
#np.savetxt("eventos.csv",eventos, delimiter=',')
# Crear carpeta ./histograma/ y guardar
try:
    os.mkdir('./histograma')  # Accedo si existe
except:
    pass
os.chdir('./histograma')

np.savetxt("cuentas.csv", cuentas, fmt='%i', delimiter=',')

print(len(cuentas))


def correlacion(dataPath):
    data = np.loadtxt(dataPath, delimiter=',')
    data = data[:]

    autocorre = np.correlate(data[:, 1], data[:, 1], mode="same")

    plt.plot(data[:, 0], data[:, 1])
    plt.figure()
    plt.plot(autocorre)
    plt.xlabel("t[s]")
    plt.ylabel('Amp[V]')
    plt.show()


# %%
### Datos ###
# os.chdir(r'D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Conteo de fotones\Dia 3\Grupo v3\histograma')
# carga cuentas.csv. Guardar con savetxt
rawData = np.loadtxt('cuentas.csv', delimiter=',')
# Acá podés eliminar cuentas muy altas, producto de malas mediciones
data = rawData[rawData < 20]
data.sort()
###
### Histograma ###
plt.figure(figsize=(10, 6))
hist, bins = np.histogram(data, bins=np.arange(np.max(data)), density=False)
bins = bins[:-1]
###
plt.bar(bins, hist, width=0.1, label='Datos')
# Configuración
plt.grid()
### Ejes ###
plt.axis('tight')
# plt.xlim((-1, 10))
# plt.ylim((0, 0.25))
plt.xlabel('Numero de eventos', fontsize=22)
plt.ylabel('Frecuencias relativas', fontsize=22)
plt.tick_params(labelsize=20)
###
plt.legend(loc=0, fontsize=20)
plt.savefig('histograma.png', bbox_inches='tight')
# %%
### Ajuste ###
### Fuciones de ayuda ###
# os.chdir(path)

# carga cuentas.csv. Guardar con savetxt
rawData = np.loadtxt('cuentas.csv', delimiter=',')
# Acá podés eliminar cuentas muy altas, producto de malas mediciones
data = rawData[rawData < 20]
data.sort()
### Histograma ###
hist, bins = np.histogram(data, bins=np.arange(np.max(data)), density=True)
bins = bins[:-1]


def poissonPDF(j, lambd): return (lambd**j) * np.exp(-lambd) / factorial(j)


def bePDF(j, lambd): return np.power(lambd, j) / np.power(1+lambd, 1+j)


###

pPoisson, pconv = opt.curve_fit(poissonPDF, bins, hist, p0=3)
pBE, pconv = opt.curve_fit(bePDF, bins, hist, p0=3)
# with open('p-valor', 'w+') as f:
#     f.write("Poisson p-value: {}\n".format(stats.chisquare(hist,
#             poissonPDF(bins, pPoisson), ddof=1)[1]))
#     f.write(
#         "BE p-value: {}".format(stats.chisquare(hist, bePDF(bins, pBE), ddof=1)[1]))
###

### Ploteo ###

width = 24 / 2.54  # 24cm de ancho
figSize = (width * (1+np.sqrt(5)) / 2, width)  # relación aurea

# Figura 1
plt.figure(figsize=figSize)
# plt.plot(poissonPDF(bins, pPoisson), 'r^',
#          label='Poisson', markersize=10)  # azul
plt.plot(bePDF(bins, pBE), 'go', label='BE', markersize=10)  # verde
plt.bar(bins, hist, width=0.3, label='Datos')
# Configuración
plt.grid()
### Ejes ###
plt.axis('tight')
# plt.xlim((-1, 10))
# plt.ylim((0, 0.25))
plt.xlabel('Numero de eventos', fontsize=22)
plt.ylabel('Frecuencias relativas', fontsize=22)
plt.tick_params(labelsize=20)
###
### Texto a agregar ###
text = 'Poisson\n'
text += r'$<n> = {0:.2f}$'.format(pPoisson[0])
text += '\n'
# text += r'$\chi^2_{{ \nu = {0} }} = {1:.3f}$'.format(hist.size, "poissonChisq")
text += '\n\n'
text += 'BE\n'
text += r'$<n> = {0:.2f}$'.format(pBE[0])
text += '\n'
# text += r'$\chi^2_{{ \nu = {0} }} = {1:.2f}$'.format(hist.size, "beChisq")
plt.text(0.7, 0.3, text, transform=plt.gca().transAxes, fontsize=14)
######
plt.legend(loc=0, fontsize=20)
plt.savefig('histograma.png', bbox_inches='tight')

# Figura 2
plt.figure(2, figsize=figSize)
plt.plot(np.log(poissonPDF(bins, pPoisson)), 'r^',
         label='Log(Poisson)', markersize=10)  # azul
plt.plot(np.log(bePDF(bins, pBE)), 'go',
         label='Log(BE)', markersize=10)  # verde
plt.plot(bins, np.log(hist), 'bd', label='Log(Datos)', markersize=10)
#### Configuración ####
plt.grid()
### Ejes ###
plt.axis('tight')
# plt.xlim((-1, 10))
# plt.ylim((-5, -1))
plt.xlabel('Numero de eventos', fontsize=22)
plt.ylabel('Log(Frecuencias relativas)', fontsize=22)
plt.tick_params(labelsize=20)
###
plt.legend(loc=0, fontsize=20)
########
plt.savefig('log_histograma.png', bbox_inches='tight')
