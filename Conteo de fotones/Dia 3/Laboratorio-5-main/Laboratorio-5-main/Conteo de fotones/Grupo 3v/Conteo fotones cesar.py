# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:14:43 2021

@author: Publico
"""

import visa
import numpy as np
import time
import matplotlib.pyplot as plt

from instrumental import TDS1002B
from scipy.signal import find_peaks

# inicializo comunicacion con equipos
rm = visa.ResourceManager()
#lista de dispositivos conectados, para ver las id de los equipos
print(rm.list_resources())

#osciloscopio
osci = TDS1002B('USB0::0x0699::0x0363::C108012::INSTR')
#osci.get_time()
#osci.set_time(scale = 1e-3)
#osci.set_channel(1,scale = 0.05)
#%%
screens = []
for i in range(500):
    tiempo, data = osci.read_data(channel = 1)
    screens.append(data)
#    plt.figure()
#    plt.plot(tiempo,data)
#    plt.xlabel('Tiempo [s]')
#    plt.ylabel('Voltaje [V]')
    print(i+1)
np.savez("Medicion1.npz", screens=np.concatenate(screens))
#%%
import os
os.chdir(r'C:\Users\Publico.LABORATORIOS\Desktop\Grupos Labo estudiantes\Grupo 3v')
plt.figure()
plt.plot(tiempo,data)
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')   
np.savetxt("data_reliminar_100ohm 25ns.txt",[tiempo,data],delimiter=',')