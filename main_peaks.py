#%%%

import numpy as np
import matplotlib.pyplot as plt
import Telecotoolbox as ttb
from sim import Sim, Fibra
import solver
import scipy.constants as const
import time

#---Parámetros de la simulación---
puntos = 2**10
Tmax   = 10000 #ps
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
ancho = 30
#Para 8 W se obtiene longitud de coeherencia inf

print("Conversion de lambda a freq: ")
lambda_shift = 0.00030 #nm
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
#---Gráficos y extracción de picos robusta (reemplaza la sección anterior) ---
import numpy as np
from scipy.signal import find_peaks

plt.close('all')

N = pulso_0.size
dt = sim.paso_t                       # [ps]
f0 = const.c / (lambda0 * 1e-9)       # [Hz] central

# Ejes de frecuencia: unshifted (fft ordering) y shifted (fftshift ordering)
freqs_un = np.fft.fftfreq(N, d=dt)            # [1/ps]
freqs_sh  = np.fft.fftshift(freqs_un)         # [1/ps]

# Convertir a Hz (1/ps -> Hz) y a lambda (nm)
freqs_un_Hz = f0 + freqs_un * 1e12
freqs_sh_Hz = f0 + freqs_sh * 1e12
lambda_un = const.c / freqs_un_Hz * 1e9
lambda_sh = const.c / freqs_sh_Hz * 1e9

# Espectro de entrada en ambas convenciones (potencia)
spec_un = np.abs(np.fft.fft(pulso_0))**2
spec_sh = np.abs(np.fft.fftshift(np.fft.fft(pulso_0)))**2

# Elegir convención que coloca el pico central más cerca de lambda0
peak_un = lambda_un[np.argmax(spec_un)]
peak_sh = lambda_sh[np.argmax(spec_sh)]
use_shift = (abs(peak_sh - lambda0) <= abs(peak_un - lambda0))

if use_shift:
    freqs_used = freqs_sh
    lambda_axis = lambda_sh
    entrada_spec = spec_sh
    AW_spec = np.fft.fftshift(AW, axes=1)   # AW debe alinearse con esta convención
else:
    freqs_used = freqs_un
    lambda_axis = lambda_un
    entrada_spec = spec_un
    AW_spec = AW.copy()

# Normalizar
entrada_spec = entrada_spec / entrada_spec.max()
salida_spec = np.abs(AW_spec[-1, :])**2
salida_spec = salida_spec / salida_spec.max()

# Ordenar por lambda creciente y remuestrear a grid lineal en nm (para separaciones uniformes en nm)
sort_idx = np.argsort(lambda_axis)
lambda_sorted = lambda_axis[sort_idx]
lambda_lin = np.linspace(lambda_sorted.min(), lambda_sorted.max(), N)

entrada_lin = np.interp(lambda_lin, lambda_sorted, entrada_spec[sort_idx])
salida_lin  = np.interp(lambda_lin, lambda_sorted, salida_spec[sort_idx])

# Graficar entrada y salida (lambda lineal)
plt.figure(figsize=(8,4))
plt.semilogy(lambda_lin, entrada_lin, label='Entrada (norm.)')
plt.axvline(lambda0, color='k', linestyle='--', label=f'lambda0={lambda0} nm')
plt.title("Espectro de entrada")
plt.xlabel("Longitud de onda [nm]"); plt.ylabel("Espectro (norm.)")
plt.grid(True); plt.legend(); plt.tight_layout()

plt.figure(figsize=(8,4))
plt.semilogy(lambda_lin, salida_lin, label='Salida (norm.)')
plt.axvline(lambda0, color='k', linestyle='--')
plt.title("Espectro de salida")
plt.xlabel("Longitud de onda [nm]"); plt.ylabel("Espectro (norm.)")
plt.grid(True); plt.legend(); plt.tight_layout()

# Mapa espectral interpolado a lambda_lin
AW_power = np.abs(AW_spec)**2                           # shape (len(z), N)
AW_map = np.vstack([np.interp(lambda_lin, lambda_sorted, AW_power[i, sort_idx]) for i in range(AW_power.shape[0])])
AW_map /= AW_map.max()

plt.figure(figsize=(10,5))
extent = (0, L, lambda_lin.min(), lambda_lin.max())
plt.imshow(AW_map.T, aspect='auto', extent=extent, origin='lower', cmap='viridis')
plt.colorbar(label='Potencia (norm.)'); plt.xlabel('z [m]'); plt.ylabel('Longitud de onda [nm]')
plt.title('Evolución espectral vs z'); plt.tight_layout()

# Extracción de picos (por z) en eje lineal de lambda: devolver principal(s) por z
num_peaks = 3   # cuántos picos extraer por z (ajustable)
peaks_vs_z = np.full((len(z), num_peaks), np.nan)

for iz in range(len(z)):
    row = AW_map[iz, :]
    # detectar picos locales robustamente (threshold relativo)
    peaks_idx, _ = find_peaks(row, height=(row.max()*0.05,))
    if peaks_idx.size > 0:
        # tomar los N mayores picos
        top = peaks_idx[np.argsort(row[peaks_idx])[-num_peaks:]]
        top_sorted = np.sort(top)  # orden por lambda ascendente
        # rellenar desde la derecha (mayor intensidad primero)
        vals = row[top_sorted]
        # si hay menos picos que num_peaks, rellenar los primeros con NaN
        k = len(top_sorted)
        peaks_vs_z[iz, :k] = lambda_lin[top_sorted][:k]

# Sobreponer los picos detectados en el mapa
for k in range(num_peaks):
    col = 'C'+str(k)
    valid = ~np.isnan(peaks_vs_z[:, k])
    plt.plot(z[valid], peaks_vs_z[valid, k], '.', color=col, markersize=4, label=f'Pico {k+1}' if k==0 else None)

plt.legend()
plt.savefig("nlse_spectral_map_with_peaks.png")
plt.show()

# Debug infos
print(f"Debug: chosen shift convention = {'fftshift' if use_shift else 'no shift'}")
print("Entrada: pico (nm) =", lambda_lin[np.argmax(entrada_lin)])
print("Salida (final): pico (nm) =", lambda_lin[np.argmax(salida_lin)])
