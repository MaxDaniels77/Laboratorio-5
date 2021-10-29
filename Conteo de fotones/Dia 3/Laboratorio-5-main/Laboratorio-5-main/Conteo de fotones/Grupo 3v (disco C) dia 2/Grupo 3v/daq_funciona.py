import matplotlib.pyplot as plt
import numpy as np
import nidaqmx
import math
import time


#para saber el ID de la placa conectada (DevX)
system = nidaqmx.system.System.local()
for device in system.devices:
    print(device)

#para setear (y preguntar) el modo y rango de un canal analógico
with nidaqmx.Task() as task:
    ai_channel = task.ai_channels.add_ai_voltage_chan("Dev2/ai1",max_val=10,min_val=-10)
    print(ai_channel.ai_term_cfg)    
    print(ai_channel.ai_max)
    print(ai_channel.ai_min)	
	

## Medicion por tiempo/samples de una sola vez
def medir(duracion, fs):
    cant_puntos = duracion*fs    
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        task.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = modo)
        task.timing.cfg_samp_clk_timing(fs,samps_per_chan = int(cant_puntos),sample_mode = nidaqmx.constants.AcquisitionType.FINITE)
        data=[]
        total = 0
        med = 0
        while med < cant_puntos:
            task.start()
            time.sleep(0.1)
            datos_temp = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)  
            for i in range(len(datos_temp)):
                data[med+i] = datos_temp[i]
            total = total + len(datos_temp)
            t1 = time.time()
            med = med + len(datos_temp)
            task.wait_until_done(timeout=1/fs)
            print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos_temp), total, total/(t1-t0))) 

        #saco los nan
    data = [value for value in data if not math.isnan(value)]
    #np.savetxt(filename+'.txt', np.transpose([data]))
    return data


duracion = 0.01 #segundos
fs = 400e3 #Frecuencia de muestreo
y = medir(duracion, fs)
plt.plot(y)
plt.grid()
plt.show()


## Medicion continua
data_total=[]
fs = 250000 #Frecuencia de muestreo
task = nidaqmx.Task()
modo= nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
task.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = modo)
task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
task.start()
t0 = time.time()
total = 0
med = 0
data = []
for i in range(50):
#    time.sleep(0.1)
    datos_temp = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)           
    print(datos_temp)
#    total = total + len(datos)
    t1 = time.time()
#    data[med+i] = datos_temp[i]
#    print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos), total, total/(t1-t0)))    
task.stop()
task.close()

plt.plot(data_total)
plt.grid()
plt.show()

#%%

def medicion_corta(fs, tiempo_medicion): #tiempo en segundos
    #devuelve la data cruda
    cantidad_mediciones = int(tiempo_medicion*fs)
    data = np.zeros(cantidad_mediciones*2)
    data[:] = np.nan
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        task.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = modo, max_val=10,min_val=-10)
        task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
        task.start()
        t0 = time.time()
        total = 0
        med = 0
        while med < cantidad_mediciones:
            time.sleep(0.1)
            datos_temp = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)  
            for i in range(len(datos_temp)):
                data[med+i] = datos_temp[i]
            total = total + len(datos_temp)
            t1 = time.time()
            med = med + len(datos_temp)
            print("%2.3fs %d %d %2.3f" % (t1-t0, len(datos_temp), total, total/(t1-t0))) 

        #saco los nan
    data = [value for value in data if not math.isnan(value)]
    #np.savetxt(filename+'.txt', np.transpose([data]))
    return data

duracion = 10 #segundos
fs = 400e3 #Frecuencia de muestreo
y = medicion_corta(fs, duracion)
plt.plot(y)
plt.grid()
plt.show()
