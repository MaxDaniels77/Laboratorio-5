import visa
import numpy as np
import time
import matplotlib.pyplot as plt

# inicializo comunicacion con equipos
rm = visa.ResourceManager()
#lista de dispositivos conectados, para ver las id de los equipos
rm.list_resources()

#inicializo generador de funciones
osci = rm.open_resource('USB0::0x0699::0x0363::C065087::INSTR')
fungen = rm.open_resource('USB0::0x0699::0x0346::C034165::INSTR')


# generador

#fungen.query('*IDN?')
##le pregunto la freq
#fungen.query('FREQ?')
##le seteo la freq
#fungen.write('FREQ 1000')
#fungen.query('FREQ?')
##le pregunto la amplitud
#fungen.query('VOLT?')
##le seteo la amplitud
#fungen.write('VOLT 2')
#fungen.query('VOLT?')
##le pregunto si la salida esta habilitada
#fungen.query('OUTPut1:STATe?')
##habilito la salida
#fungen.write('OUTPut1:STATe 1')
#fungen.query('OUTPut1:STATe?')
##le pregunto la impedancia de carga seteada
#fungen.query('OUTPUT1:IMPEDANCE?')
#
## osci
#
##le pregunto su identidad
#osci.query('*IDN?')
##le pregunto la conf del canal (1|2)
#osci.query('CH1?')
##le pregunto la conf horizontal
#osci.query('HOR?')
##le pregunto la punta de osciloscopio seteada
#osci.query('CH2:PRObe?')
#
#
#
##Seteo de canal
#channel=1
#scale = 5e-1
#osci.write("CH{0}:SCA {1}".format(channel, scale))
#osci.query("CH{0}:SCA?".format(channel))
#"""escalas Voltaje (V) ojo estas listas no son completas
#2e-3
#5e-3
#10e-3
#20e-3
#50e-3
#100e-3
#5e-2
#10e-2
#"""
#
#zero = 0
#osci.write("CH{0}:POS {1}".format(channel, zero))
#osci.query("CH{0}:POS?".format(channel))
#
#channel=2
#scale = 2e-1
#osci.write("CH{0}:SCA {1}".format(channel, scale))
#osci.write("CH{0}:POS {1}".format(channel, zero))
#
##seteo escala horizontal
#scale = 200e-6
#osci.write("HOR:SCA {0}".format(scale))
#osci.write("HOR:POS {0}".format(zero))	
#osci.query("HOR?")
#"""
#escalas temporales (s)
#10e-9
#25e-9
#50e-9
#100e-9
#250e-9
#500e-9
#1e-6
#2e-6
#5e-6
#10e-6
#25e-6
#50e-6
#"""
#%%

from instrumental import AFG3021B
from instrumental import TDS1002B
from instrumental import GPIB0

#%%osciloscopio
osci = TDS1002B('USB0::0x0699::0x0363::C065087::INSTR')
osci.get_time()
osci.set_time(scale = 250e-6)

osci.set_channel(1,scale = 1)
osci.set_channel(2,scale = 2)
tiempo1, data1 = osci.read_data(channel = 1)
tiempo2, data2 = osci.read_data(channel = 2)

plt.figure()
plt.plot(tiempo1,data1)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')

plt.figure()
plt.plot(tiempo2,data2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
#
##%%
##Guardar txt con los datos 
#datos_ch1 = np.array([tiempo1,data1])
#datos_ch2 = [tiempo2,data2]
#np.savetxt('Datos_ch1.txt',datos_ch1,delimiter=',')
#np.savetxt('Datos_ch2.txt',datos_ch2,delimiter=',')
#%%

#generador de funciones
fungen = AFG3021B(name = 'USB0::0x0699::0x0346::C034165::INSTR')
fungen.getFrequency()
fungen.setFrequency(2000)
#%% barrido de frecuencia
#frec = np.logspace(np.log10(100),np.log10(1000),10)
#
#for freq in range(100,1500,500):
##    print(freq)
#    fungen.setFrequency(freq)
#    time.sleep(0.1)
#    # 
##   tiempo, data = osci.read_data(channel = 1) 
#    
#    plt.figure()
#    plt.plot(tiempo,data)


#%% BARREDOR DE FRECUENCIAS 3000
fungen.setSin(1)
#osci.set_time(scale = 200e-6)

frec_min = 10
frec_max= 10000
step = 50
frecuencias = np.logspace(np.log10(frec_min),np.log10(frec_max),step)
val = np.zeros(len(frecuencias))

for ii, freq in enumerate(frecuencias):
    fungen.setFrequency(freq)
    osci.set_time(0.3/freq)
    time.sleep(0.1)   
    val[ii]=osci.get_pk2pk(2)
    
plt.figure()
plt.plot(frecuencias,val)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Voltaje [V]')
plt.xscale('log')
#%% DETERMINACION DE TAU 

fungen.setCuadrada(1)
fungen.setFrequency(1000)
osci.set_time(scale = 100e-6)

osci.set_channel(1,scale = 1)
osci.set_channel(2,scale = 5)
tiempo1, data1 = osci.read_data(channel = 1)
tiempo2, data2 = osci.read_data(channel = 2)

plt.figure()
plt.plot(tiempo1,data1)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')

#plt.figure()
plt.plot(tiempo2,data2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')
#%% DESCARGA GLOW

from instrumental import AFG3021B
from instrumental import GPIB0

multi = GPIB0()
fungen = AFG3021B(name = 'USB0::0x0699::0x0346::C036492::INSTR')
fungen.setRamp(1)
fungen.setFrequency(0.02)
fungen.getFrequency()
fungen.setAmplitude(3.2)
fungen.setOffset(1,1600)
#print(multi.get_voltage())

v_min = 0
v_max= 1000
step = 50
voltajes = range(v_min,v_max,1)
volt_meas = np.zeros(len(voltajes))

for ii, volt in enumerate(voltajes):
#    fungen.setAmplitude(volt*10/2500)
    time.sleep(0.5)   
    volt_meas[ii]=multi.get_voltage()
    
plt.figure()
plt.plot(voltajes,volt_meas)
plt.xlabel('V_send [V]')
plt.ylabel('V_measured [V]')
