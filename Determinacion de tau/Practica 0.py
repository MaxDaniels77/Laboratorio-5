# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Determinacion de tau')
data1 = np.loadtxt('D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Determinacion de tau/Datos_ch1.txt',delimiter=',')
data2 = np.loadtxt("D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Determinacion de tau/Datos_ch2.txt",delimiter=',')
print(data1)
plt.plot(data1[0],data1[1])
plt.plot(data2[0],data2[1])


