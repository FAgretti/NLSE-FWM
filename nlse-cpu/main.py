
import numpy as np
import matplotlib.pyplot as plt
import Telecotoolbox as ttb
from sim import Sim, Fibra
import solver
import scipy.constants as const
import time
import sys

#---Parámetros de la simulación---
puntos = 2**13
if(sys.argv[1]):
    print(sys.argv[1])
    puntos = 2**int(sys.argv[1])
Tmax   = 1000 #ps
sim = Sim(puntos, Tmax)
fs = 1/sim.paso_t

#---Parámetros de la fibra---
L = 4*8976 #m
gamma = 1.4e-3 #1/(W*m)
gamma1 = 0
alpha = 0
lambda0 = 980 #nm
betas = [-20e-3,0*10e-3,0*1e-3] #ps^n/m
fib = Fibra(L, gamma, gamma1, alpha, lambda0, betas)


#---Pulso de entrada---
Pot = 0.25 #W
#Para 8 W se obtiene longitud de coeherencia inf

#---Parametros del FWM
freqShift = np.sqrt(gamma*Pot/np.abs(betas[0])) #THz
#freqShift = 0.23*(2*np.pi)
freqFWM = freqShift/(2*np.pi)

#pulso_0 = 1.5*np.sqrt(Pot)*np.ones(puntos)*np.exp(-1j*2*np.pi*freqShift*sim.tiempo)
pump = 1*np.sqrt(Pot)*np.ones(puntos)*np.exp(-1j*2*np.pi*0*freqShift*sim.tiempo)
pulso_1 = 0.1*np.sqrt(Pot)*np.ones(puntos)*np.exp(1j*2*np.pi*freqFWM*sim.tiempo)
ruido = 0.01*np.sqrt(Pot)*np.random.normal(0,0.1,puntos)

pulso_0 = pump +pulso_1+ ruido

#---Simulación---
z_locs = 100

time0 = time.time()
#print("Tiempo inicial: ",time0)
z, AW = solver.SolveNLS(sim, fib, pulso_0, z_locs=z_locs)
print("Tiempo final: ",time.time()-time0)

# deltaKm = betas[0]*(freqShift)**2+(betas[2]/12)*(freqShift)**4
# deltaKnl = gamma*Pot

# kappa = deltaKm + deltaKnl
# if(kappa):
#     Lcoh = np.inf
# else:
#     Lcoh = (2*np.pi/kappa) #m?
# print("Longitud de coherencia = ",Lcoh)

# Linteraccion = (np.pi/(gamma*Pot))
# print("Longitud de interaccion = ",Linteraccion)

# #La longitud en la que se da la conversion total de potencia del pump al idler y signal sigue una ley de 2*pi/(2*gamma*Pot)
# #Para 16W ~ 150m
# #Para 8W ~ 300m         #Todos con gamma  = 1.4 e-3
# #Para 4W ~ 600m

# print("Frecuencia = ",freqFWM)
# #---Gráficos---
# plt.figure()

# plt.subplot(211)
# """
# plt.plot(sim.tiempo, np.abs(pulso_0)**2)

# plt.title("Pulso de entrada", fontsize=24)
# plt.subplot(212)
# """
# plt.plot(sim.freq, np.abs(ttb.TFopt(pulso_0)**2))

# plt.title("Espectro de entrada", fontsize=16)
# plt.xlabel("Frecuencia [THz]", fontsize=16)
# plt.ylabel("Potencia [W]", fontsize=16)
# plt.xlim(-0.5,0.5)

# #plt.figure()
# """
# plt.subplot(211)
# plt.plot(sim.tiempo, np.abs(AT[-1])**2)
# plt.title("Pulso de salida",fontsize=24)
# """
# plt.subplot(212)

# plt.plot(sim.freq, np.abs(AW[-1])**2)
# plt.title("Espectro de salida",fontsize=16)
# plt.xlabel("Frecuencia [THz]", fontsize=16)
# plt.ylabel("Potencia [W]", fontsize=16)
# plt.xlim(-0.5,0.5)

# plt.subplots_adjust(left=0.1, right=0.9, 
#                     top=0.9, bottom=0.1, 
#                     wspace=0.4, hspace=0.6)

# plt.savefig("nlse2048CPUsalida.png")

# plt.figure()
# #plt.imshow(np.abs(np.fft.fftshift(ttb.TFopt(AT)))**2, aspect='auto', extent=[np.min(sim.freq), np.max(sim.freq),0,L])
# AW = np.sqrt(Pot)*(np.fft.fftshift(AW,axes=1)/np.max(np.abs(AW)))
# plt.imshow((np.abs(AW)**2), aspect='auto', extent=[np.min(sim.freq), np.max(sim.freq),0,L], origin='lower')
# plt.xlim(np.min(sim.freq), np.max(sim.freq))
# #plt.xlim(-2,2)
# plt.colorbar()
# plt.title("Espectro de salida en función de z")

# plt.savefig("nlse2048CPUimfreq.png")
# plt.figure()
# plt.imshow(np.abs(AT)**2, aspect='auto', origin='lower', extent=[sim.tiempo[0], sim.tiempo[-1],0,L])
# #plt.ylim([0, Tmax])
# plt.xlim([-20, 20])
# plt.colorbar()
# plt.title("Evolución temporal del pulso")

# plt.savefig("nlse2048CPUimtime.png")

# plt.figure()
# ind = np.where(np.abs(AW[0,:])>0.09*np.sqrt(Pot))
# print(ind[0])
# plt.plot(np.linspace(0,L,z_locs),np.abs(AW[:,ind[0][0]])**2, label = "Idler")
# plt.plot(np.linspace(0,L,z_locs),np.abs(AW[:,ind[0][1]])**2, label = "Pump")
# plt.plot(np.linspace(0,L,z_locs),np.abs(AW[:,(2*ind[0][1]-ind[0][0])])**2, label = "Signal")
# plt.plot(np.linspace(0,L,z_locs),np.abs(AW[:,(2*ind[0][1]-ind[0][0])])**2 + np.abs(AW[:,ind[0][0]])**2, label = "Signal + idler")
# #plt.plot(np.linspace(0,L,z_locs),np.abs(AW[:,4096])**2, label = "pump")
# suma = np.zeros(z_locs)
# #for i in range(len(ind[0])):
# #    suma = suma + np.abs(AW[:,ind[0][i]])**2
# #suma = suma - np.abs(AW[:,128])**2
# #plt.plot(np.linspace(0,L,z_locs),suma, label = "Productos no lineales")
# plt.grid()
# plt.legend(loc = "upper right", fontsize=16)
# #vector del FWM
# """
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# textstr = '\n'.join((
#     r'$\Delta k_{m} = %.3e$ $[\frac{1}{m}]$' % (deltaKm, ),
#     r'$\Delta k_{nl} = %.3e$ $[\frac{1}{m}]$' % (deltaKnl, ),
#     r'$\kappa = %.3e$ $[\frac{1}{m}]$' % (kappa, ),
#     r'$L_{coh} = %.3e$ [m]' % (Lcoh, ),
#     r'$L_{interaccion} = %.3e$ [m]' % (Linteraccion, )))


# plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)
# """
# plt.title("Potencia del FWM en función de z", fontsize=20)
# plt.xlabel("z [m]", fontsize=20)
# plt.ylabel("Potencia [W]", fontsize=20)

# plt.savefig("nlse2048CPU.png")

# plt.show()

# # %%
