# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 23:10:29 2021

@author: cyber
"""

import matplotlib.pyplot as plt
import numpy as np
import os 

os.chdir('D:\Documentos\Documentos-facultad\Labo 5\Laboratorio-5\Descarga glow')
#%% CURVA DE HISTERESIS
# data0 = np.loadtxt("D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Descarga glow/curva1.txt",delimiter=",")
# fig = plt.figure(figsize=(8,5 ))
# plt.rc('font', family='serif', size=18)
# # plt.clf()
# ax = fig.add_subplot()

# ax.plot(data1[0]/150,data1[1]*1018, '.', label="asd")


# ax.set_title('Corriente vs Voltaje')
# ax.ticklabel_format(axis='x',style='sci',scilimits=(1,4))

# ax.set_xlabel('Corriente [A]')
# ax.set_ylabel('Voltaje [V]')
# plt.legend()
# plt.tight_layout()  
# plt.subplots_adjust(bottom=0.114,left=0.124, right=0.996, top=0.936)
# plt.show()
#%% CURVA DE I VS V
''' Es necesario acomodar los datos nuevos que tomamos para poder tener un grafico adecuado 
que reuna toda la informacion para esto parto de la informacion que transcribio a mano lean'''

data1 = np.loadtxt("D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Descarga glow/curva1.txt",delimiter=",")
data2 = np.loadtxt("D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Descarga glow/curva2.txt",delimiter=",")
data3 = np.genfromtxt(r"D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Descarga glow/Control I curva V vs I.csv",delimiter=";",skip_header=1)
data_histeresis = np.genfromtxt(r"D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Descarga glow/histeresis.csv.csv",delimiter=";",skip_header=1)
fig = plt.figure(figsize=(8,5 ))
plt.rc('font', family='serif', size=18)
# plt.clf()
ax = fig.add_subplot()

ax.plot(data1[0]/150,data1[1]*1018, '.', label="Medicion 1")
ax.plot(data2[0]/150,data2[1]*1018, '.', label="Medicion 2")
ax.plot(data3[:,6]/(1000*150),data3[:,4]*1018/1000, '.', label="Medicion 3")
# ax.plot(data_histeresis[:,6]/(1000*150),data_histeresis[:,4]*1018/1000, '.', label="Medicion 3")

ax.set_title('Corriente vs Voltaje')
ax.ticklabel_format(axis='x',style='sci',scilimits=(1,4))

ax.set_xlabel('Corriente [A]')
ax.set_ylabel('Voltaje [V]')
plt.legend()
plt.tight_layout()  
plt.subplots_adjust(bottom=0.114,left=0.124, right=0.996, top=0.936)
plt.show()
plt.savefig("Ploteo de datos.png")
#%%  CURVAS DE PASCHEN

data4= np.genfromtxt(r"D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Descarga glow/Paschen 2.0mbar.csv",delimiter=";",skip_header=1)
data5= np.genfromtxt(r"D:/Documentos/Documentos-facultad/Labo 5/Laboratorio-5/Descarga glow/Paschen a 3cm.csv",delimiter=";",skip_header=1)

# p4 = data4[:,0]/1013 # pasado a atmosferas
# d4=data4[:,2]/100 # pasado a metros
# v4=data4[:,6]*(1018/1000)

p4 = data4[:,0]
d4=data4[:,2]#
v4=data4[:,6]*(1018/1000)

# p5 = data5[:,0]/1013 # pasado a atmosferas
# d5=data5[:,2]/100 # pasado a metros
# v5=data5[:,6]*(1018/1000)

p5 = data5[:,0] # pasado a atmosferas
d5=data5[:,2] # pasado a metros
v5=data5[:,6]*(1018/1000)

plt.plot(p4*d4,v4,".")
plt.plot(p5*d5,v5,".")

#%% AJUSTE LEY DE PASCHEN 2.0mbar
from scipy.optimize import curve_fit
def ajuste(x,a,b): return (a*x)/(np.log(x)+b)

x1 = p4*d4
y1 = v4
# y1[2]=355

#Errores
error_p4=data4[:,1]
error_d4=data4[:,3]

error_pd4 = np.sqrt((error_p4*d4)**2+(error_d4*p4)**2)
error_v4 = data4[:,7] *(1018/1000)


p0 = [10e6,12.8]
params4, mcov4 = curve_fit(ajuste, x1, y1, p0=p0)  # fitteo
sigmas4 = np.sqrt(np.diag(mcov4))

# fig, ax = plt.subplots()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
# ax.plot(x1,y1,".",label='Datos')
ax.errorbar(x1,y1,error_v4 ,error_pd4,fmt="+" ,label="Datos")
# plt.rcParams['axes.facecolor'] = '#f6f6f6'
# fig.set_facecolor('#f6f6f6')

ax.plot(np.linspace(x1.min(),x1.max(),1000),ajuste(np.linspace(x1.min(),x1.max(),1000),*params4),label='ajuste')
# plt.plot( np.linspace(x.min(),x.max(),100),f(np.linspace(x.min(),x.max(),100),*params))

ax.set_title("Ajuste curvas de Paschen",fontsize=18)
ax.ticklabel_format(axis='x',style='sci',scilimits=(1,4))

ax.set_xlabel('P*D',fontsize=15)
ax.set_ylabel('Voltaje [V]',fontsize=15)
# ax.xticks(fontsize=13)
# ax.yticks(fontsize=13)
# ax.legend(fontsize=15)
ax.grid(True)
# ax.set_xscale('log')
# plt.yscale('log')
plt.legend()
plt.tight_layout()  
plt.show()

plt.savefig('Ajuste curva de paschen 2mbar.png')

print(params4,sigmas4)
x_ajuste1 = np.linspace(x1.min(),x1.max(),1000)
y_ajuste1 =ajuste(np.linspace(x1.min(),x1.max(),1000),*params4)
#%% AJUSTE LEY DE PASCHEN 3cm
from scipy.optimize import curve_fit
def ajuste(x,a,b): return (a*x)/(np.log(x)+b)

x1 = p5*d5
y1 = v5

#errores
error_p5=data5[:,1]
error_d5=data5[:,3]

error_pd5 = np.sqrt((error_p5*d5)**2+(error_d5*p5)**2)
error_v5 = data5[:,7] *(1018/1000)



p0 = [10e6,12.8]
params5, mcov5 = curve_fit(ajuste, x1, y1, p0=p0)  # fitteo
sigmas5 = np.sqrt(np.diag(mcov5))

# fig, ax = plt.subplots()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
ax.plot(x1,y1,".",label='Datos')
ax.errorbar(x1,y1,error_v5 ,error_pd5,fmt="+" ,label="Datos")

# plt.rcParams['axes.facecolor'] = '#f6f6f6'
# fig.set_facecolor('#f6f6f6')

ax.plot(np.linspace(x1.min(),x1.max(),1000),ajuste(np.linspace(x1.min(),x1.max(),1000),*params5),label='ajuste')
# plt.plot( np.linspace(x.min(),x.max(),100),f(np.linspace(x.min(),x.max(),100),*params5))

ax.set_title("Ajuste curva de paschen",fontsize=18)
ax.ticklabel_format(axis='x',style='sci',scilimits=(1,4))

ax.set_xlabel('PD',fontsize=15)
ax.set_ylabel('Voltaje [V]',fontsize=15)
# ax.xticks(fontsize=13)
# ax.yticks(fontsize=13)
# ax.legend(fontsize=15)
ax.grid(True)
# ax.set_xscale('log')
# plt.yscale('log')
plt.legend()    
plt.tight_layout()  
plt.show()

plt.savefig('Ajuste Rabi.png')

print(params5,sigmas5)
x_ajuste1 = np.linspace(x1.min(),x1.max(),1000)
y_ajuste1 =ajuste(np.linspace(x1.min(),x1.max(),1000),*params5)



#%%AJUSTE USANDO MINIMIZE 
from scipy.optimize import minimize 

def ajuste2(x,p): return (p[0]*x)/(np.log(x)+p[1]) # modelo

x1 = p4*d4
y1 = v4
# y1[2]=355

#Errores
error_p4=data4[:,1]
error_d4=data4[:,3]

error_pd4 = np.sqrt((error_p4*d4)**2+(error_d4*p4)**2)
error_v4 = data4[:,7] *(1018/1000)

p =params4
pp0 = np.concatenate((p,x1))

def chi2(pp): 
    xfit = pp[2:len(pp)]
    p = pp[0:2]
    chi = sum(((y1-ajuste2(xfit,p))/error_v4)**2 + ((x1-xfit)/error_pd4)**2)
    return chi
                         
# print(chi2(pp0))
minimizada = minimize(chi2,pp0,method='BFGS')

# fig, ax = plt.subplots()
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot()
# ax.plot(x1,y1,".",label='Datos')
ax.errorbar(x1,y1,error_v4 ,error_pd4,fmt="+" ,label="Datos")
# plt.rcParams['axes.facecolor'] = '#f6f6f6'
# fig.set_facecolor('#f6f6f6')

ax.plot(np.linspace(x1.min(),x1.max(),1000),ajuste2(np.linspace(x1.min(),x1.max(),1000),minimizada.x[0:2]),label='Ajuste usando minimize')
ax.plot(np.linspace(x1.min(),x1.max(),1000),ajuste(np.linspace(x1.min(),x1.max(),1000),*params4),label='Ajuste usando curve_fit')
ax.set_ylim(330,525)
# plt.plot( np.linspace(x.min(),x.max(),100),f(np.linspace(x.min(),x.max(),100),*params))

ax.set_title("Ajuste curvas de Paschen",fontsize=18)
ax.ticklabel_format(axis='x',style='sci',scilimits=(1,4))

ax.set_xlabel('P*D[mBar*cm]',fontsize=15)
ax.set_ylabel('Voltaje [V]',fontsize=15)
# ax.xticks(fontsize=13)
# ax.yticks(fontsize=13)
# ax.legend(fontsize=15)
# ax.grid(True)
# ax.set_xscale('log')
# plt.yscale('log')
plt.legend()
plt.tight_layout()  
plt.subplots_adjust(bottom=0.114,left=0.124, right=0.996, top=0.936)
plt.show()
plt.savefig('ajuste paschen.png')
#%%
a = 585.29088229
b = 1.51920282
error_a = 0.05*a
error_b = 0.05*b



def error(x): return np.sqrt(((x/(np.log(x)+b))*(error_a))**2+ (error_b)**2 * ((a*x)**2) * (1/(np.log(x)+b)**2)**2)
#minimize
print(error(0.5949949949949951))
#curve_fit
a = params4[0]
b = params4[1]
error_a = sigmas4[0]
error_b = sigmas4[1]
print(error(0.5312312312312313))