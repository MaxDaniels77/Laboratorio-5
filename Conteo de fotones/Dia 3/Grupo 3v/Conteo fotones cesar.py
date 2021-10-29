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
for i in range(1):
    tiempo, data = osci.read_data(channel = 1)
    screens.append(data)
#    plt.figure()
#    plt.plot(tiempo,data)
#    plt.xlabel('Tiempo [s]')
#    plt.ylabel('Voltaje [V]')
    print(i+1)
np.savez("pantalla autocorrelacion 250ms (ultima que hice)", screens=np.concatenate(screens))
np.savetxt("pantalla autocorrelacion 250ms (ultima que hice)",[tiempo,data],delimiter=',')
#%%
import os
os.chdir(r'D:\Alumnos\Laboratorio-5-main\Conteo de fotones\Grupo v3 dia 2 disco d\Grupo v3')
plt.figure()
plt.plot(tiempo,data)
plt.grid(True)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')   
#%%
np.savetxt("1000 pantallas 1 us 10mv (2)",[tiempo,data],delimiter=',')