#%%%

import numpy as np
import matplotlib.pyplot as plt
import Telecotoolbox as ttb
from sim import Sim, Fibra
import solver
import scipy.constants as const
import time

#---Parámetros de la simulación---
puntos = 2**14
Tmax   = 200 #ps
sim = Sim(puntos, Tmax)
fs = 1/sim.paso_t

#---Parámetros de la fibra---
L = 40000 #m
gamma = 1.4e-3 #1/(W*m)
gamma1 = 0
alpha = 0
lambda0 = 1539.8 #nm
betas = [-20e-3,0*10e-3,0*1e-3] #ps^n/m
fib = Fibra(L, gamma, gamma1, alpha, lambda0, betas)


#---Pulso de entrada---
Pot = ttb.dBmtoW(23.74+0)
print("Potencia de entrada = ", Pot, "W")
ancho = 25
#Para 8 W se obtiene longitud de coeherencia inf

print("Conversion de lambda a freq: ")
lambda_shift = 0.25 #nm
freqShift = const.c / (lambda0 + lambda_shift) - const.c / (lambda0 - lambda_shift)
print("Frecuencia de desplazamiento = ", freqShift, "Hz")

#---Parametros del FWM
#freqShift = np.sqrt(gamma*Pot/np.abs(betas[0])) #THz
#freqShift = 0.05*(2*np.pi)
freqFWM = freqShift#/(2*np.pi)

#pulso_0 = 1.5*np.sqrt(Pot)*np.ones(puntos)*np.exp(-1j*2*np.pi*freqShift*sim.tiempo)
# pump = 1*np.sqrt(Pot)*np.ones(puntos)*np.exp(-1j*2*np.pi*0*freqShift*sim.tiempo)
#quiero que el pump tenga un ancho que pueda modificar
# pump = pump * np.exp(-sim.tiempo**2/(2*(0.1*100)**2)) #ps

pump = ttb.SuperGauss(sim.tiempo, Pot, ancho, 0, 0, 1)*np.exp(1j*2*np.pi*-freqShift*sim.tiempo)
pump2 = ttb.SuperGauss(sim.tiempo, Pot, ancho, 0, 0, 1)*np.exp(1j*2*np.pi*freqShift*sim.tiempo)
pulso_1 = 0.1*np.sqrt(Pot)*np.ones(puntos)*np.exp(1j*2*np.pi*freqFWM*sim.tiempo)*0
ruido = 0.01*np.sqrt(Pot)*np.random.normal(0,0.1,puntos)

pulso_0 = pump +pulso_1+ ruido + pump2

#---Simulación---
z_locs = 100

time0 = time.time()
#print("Tiempo inicial: ",time0)
z, AW = solver.SolveNLS(sim, fib, pulso_0, z_locs=z_locs)
AT = ttb.IFTopt(AW)
print("Tiempo final: ",time.time()-time0)

deltaKm = betas[0]*(freqShift)**2+(betas[2]/12)*(freqShift)**4
deltaKnl = gamma*Pot


kappa = deltaKm + deltaKnl
if kappa == 0:
    Lcoh = np.inf
else:
    Lcoh = (2*np.pi/kappa) #m?
print("Longitud de coherencia = ",Lcoh)

Linteraccion = (np.pi/(gamma*Pot))
print("Longitud de interaccion = ",Linteraccion)

#La longitud en la que se da la conversion total de potencia del pump al idler y signal sigue una ley de 2*pi/(2*gamma*Pot)
#Para 16W ~ 150m
#Para 8W ~ 300m         #Todos con gamma  = 1.4 e-3
#Para 4W ~ 600m

print("Frecuencia = ",freqFWM)
#---Gráficos---

#%%%
plt.close('all')

# Parámetros para eje espectral
N = pulso_0.size
dt = sim.paso_t                       # [ps]
# eje de frecuencia en ciclos por ps (1/ps == THz), ordenado como fftshift
freq_axis = np.fft.fftshift(np.fft.fftfreq(N, d=dt))   # [1/ps]
# frecuencia absoluta en Hz (convertir 1/ps -> Hz multiplicando por 1e12)
f0 = const.c / (lambda0 * 1e-9)                       # [Hz]
freq_Hz = f0 + freq_axis * 1e12                       # [Hz]
# eje de longitudes de onda en nm (no lineal respecto a freq)
lambda_axis = 1510.85+const.c / (54*freq_Hz) * 1e9                 # [nm]

# Espectros: usar FFT de numpy con fftshift para alinearlo con freq_axis
entrada_spec = np.abs(np.fft.fftshift(np.fft.fft(pulso_0)))**2
entrada_spec /= entrada_spec.max()

# AW puede venir en diferente ordering; aplicar fftshift por columnas para alinear con freq_axis
# Se asume AW shape = (len(z), N)
AW_shifted = np.fft.fftshift(AW, axes=1)
salida_spec = np.abs(AW_shifted[-1, :])**2
salida_spec /= salida_spec.max()

# Ordenar por longitud de onda creciente para graficar vs lambda
sort_idx = np.argsort(lambda_axis)
lambda_sorted = lambda_axis[sort_idx]
entrada_sorted = entrada_spec[sort_idx]
salida_sorted  = salida_spec[sort_idx]

# Remuestrear a eje lineal en nm para separación uniforme (opcional pero recomendado)
lambda_lin = np.linspace(lambda_sorted.min(), lambda_sorted.max(), N)
entrada_interp = np.interp(lambda_lin, lambda_sorted, entrada_sorted)
salida_interp  = np.interp(lambda_lin, lambda_sorted, salida_sorted)

lambda_lin = np.fft.fftshift(lambda_lin)

# Gráfica espectro de entrada
plt.figure(figsize=(8,4))
plt.semilogy(lambda_lin, entrada_interp, label='Espectro entrada (norm.)')
plt.title("Espectro de entrada", fontsize=16)
plt.xlabel("Longitud de onda [nm]", fontsize=14)
plt.ylabel("Potencia (norm.)", fontsize=14)
plt.xlim(lambda_lin.min(), lambda_lin.max())
plt.grid(True)
plt.legend()
plt.tight_layout()

# Gráfica espectro de salida
plt.figure(figsize=(8,4))
plt.semilogy(lambda_lin, salida_interp)#, label='Espectro salida')
plt.title("Espectro de salida", fontsize=16)
plt.xlabel("Longitud de onda [nm]", fontsize=14)
plt.ylabel("Potencia relativa [dB]", fontsize=14)
plt.xlim(1538.5, 1541)
plt.yticks(ticks=[10e-10, 10e-8, 10e-6, 10e-4, 10e-2, 1], labels=['-100','-80','-60','-40','-20','0'])
plt.grid(True)
plt.legend()
plt.tight_layout()

# Mapa espectral (evolución en z) interpolado sobre lambda lineal
# Construimos matriz (len(z), N) con potencias y la interpolamos a lambda_lin
z_vec = z  # z devuelto por solver
AW_power = np.abs(AW_shifted)**2
# Interpolar cada fila en lambda_sorted -> lambda_lin
AW_map = np.vstack([np.interp(lambda_lin, lambda_sorted, AW_power[i, sort_idx]) for i in range(AW_power.shape[0])])
AW_map /= AW_map.max()

plt.figure(figsize=(10,5))
extent = (0, L, lambda_lin.min(), lambda_lin.max())   # z desde 0..L, lambda range
plt.imshow(AW_map.T, aspect='auto', extent=extent, origin='lower', cmap='viridis')
plt.xlabel('z [m]', fontsize=14)
plt.ylabel('Longitud de onda [nm]', fontsize=14)
plt.title('Evolución espectral en función de z', fontsize=16)
plt.colorbar(label='Potencia (norm.)')
plt.tight_layout()
plt.savefig("nlse_spectral_map_wavelength.png")

plt.show()

# %%
