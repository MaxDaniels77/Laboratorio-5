# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:02:55 2021

@author: Publico
"""

import visa
from instrumental import TDS1002B

#%%
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#%% Veo el codigo del osciloscopio
rm = visa.ResourceManager()
rm.list_resources()

#%% Osciloscopio
osci = TDS1002B('USB0::0x0699::0x0363::C108012::INSTR')
osci.get_time()

#%% Escalas 
"""
100 Ohm:
    ver pulso: 10ns 2mV -- Ancho: 20ns

"""

osci.set_time(scale = 10e-9)

osci.set_channel(1, scale = 0.002)

#%% Veo si trae bien la pantalla
t0, d = osci.read_data(channel = 1)
t = t0 - t0[0]
plt.plot(t, d)
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')

#%% Guardar pantalla
Datos01 = np.zeros([len(d),1])
Datos01[:,0] = t
Datos01[:,1] = d

np.savetxt("pantalla 1.csv", Datos01, delimiter = ",")

#%% Busca picos
peaks = find_peaks(-data, height = 0.001, distance = 4)
p = peaks[0]
print(len(p))

#%% Veo si los picos son los picos
pic_x = []
pic_y = []
for i in p:
    pic_x.append(t[i])
    pic_y.append(d[i])

plt.plot(t, d, color = '#0504aa', label = 'Mediciones')
plt.plot(pic_x, pic_y, ".", color = "red", label = 'Maximos')
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
plt.legend()

#%% 
plt.hist(data)

#%% Muestreo
horizontal = 10e-6    # tiempo
vertical = 5e-3       # voltaje   

osci.set_time(scale = horizontal)
osci.set_channel(1, scale = vertical)

#%%

cuadros = 500
tiempo, data = osci.read_data(channel = 1)
time = tiempo - tiempo[0]
for i in range(cuadros):
   print(i) 
   tiempo0, data0 = osci.read_data(channel = 1)
   time0 = tiempo0 - tiempo0[0]
   data = np.concatenate([data, data0])
   time = np.concatenate([time, time0])

#%% Guardar muestreo crudo
Datos02 = np.zeros([len(data),2])
Datos02[:,0] = time
Datos02[:,1] = data

import os
os.chdir(r'C:\Users\Publico.LABORATORIOS\Desktop\Grupos Labo estudiantes\Grupo 3v')
np.savetxt("medicion2.csv", Datos02, delimiter = ",")


#%%
Datos02 = np.loadtxt('medicion2.csv', delimiter = ',')
data = Datos02[0]

#%% Busca picos 


umbral = 0.001

#peaks = find_peaks(-data, height = umbral, distance = 5)
peaks = find_peaks(-data, distance = 30)
peaks_arriba = find_peaks(data, distance = 30)

#p = peaks[0]
p_v = peaks[0]
p_a = peaks_arriba[0]
p = np.concatenate([p_v, p_a])

print(len(p))   

alturas = []
for index in p:
    alturas.append(data[index])
#%% Plot 

plt.plot(time[0:500], data[0:500])


#%% Plot histograma
    
mediciones = alturas*1000
#mediciones = data    

bins = np.arange(-0.070, 0.010, 0.001)-0.01 *0

plt.hist(mediciones, bins, color='#0504aa', alpha = 0.7, rwidth = 0.85, label = 'Mediciones' )
plt.grid(True)
plt.legend()
plt.xlabel('Alturas[V]')
plt.ylabel('Apariciones')    
plt.title('Histograma')
plt.yscale("log")




