import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt
import scipy.constants as const
from sim import Sim, Fibra
import Telecotoolbox as ttb
from scipy.integrate import solve_ivp
from tqdm import tqdm
import scipy.special as sp

""" SOLVENLS
solveNLS: Función para simular la evolución con la NLSE
sim:      Parámetros de la simulación
fib:      Parámetros de la fibra
pulso_0:  Pulso de entrada
raman:    Booleano, por defecto False. Si False, se usa aproximación de pulso ancho, si True respuesta completa.
z_locs:   Int, opcional. En cuantos puntos de z (entre 0 y L) se requiere la solución. 
pbar:     Booleano, por defecto True. Barra de progreso de la simulación.
"""
def dBdz(z,B,D,w,gamma): 
    A_w = B * np.exp(D*z)
    A_t = ttb.IFTopt(A_w)
    op_nolin = ttb.TFopt( 1j*gamma*(np.abs(A_t)**2)*A_t)
    return np.exp(-D*z) * op_nolin


def SolveNLS(sim: Sim, fib: Fibra, pulso_0, z_locs=None, pbar = True):

    #Calculamos el espectro inicial, es lo que vamos a evolucionar.
    fs  = sim.fs
    sampnum = len(pulso_0)
    espectro_0 = ttb.TFopt(pulso_0)
    
    #Calculamos el operador lineal
    if fib.betas:
        D_w = 0
        for i in range(len(fib.betas)):
            D_w = D_w + 1j*fib.betas[i]/sp.factorial(i+2) * (2*np.pi*sim.freq)**(i+2)
        D_w = np.array(D_w)

    #f_B = partial(dBdz, D = D_w, w = 2*np.pi*sim.freq, gamma = fib.gamma, TR = fib.TR)
    f_B = lambda z, B: dBdz(z,B,D_w,2*np.pi*sim.freq,fib.gamma)

    #Tolerancias para integrar (Tolerancias estandar: rtol=1e-5, atol=1e-8)
    rtol = 1e-3
    atol = 1e-6

    if pbar:  # Por si queremos la barra de progreso
        with tqdm(total=fib.L, unit="m") as pbar:
        
            def dBdz_with_progress(z, B):
                pbar.update(abs(z - dBdz_with_progress.prev_z))
                dBdz_with_progress.prev_z = z
                return dBdz(z, B, D = D_w, w = 2*np.pi*sim.freq, gamma = fib.gamma)
            dBdz_with_progress.prev_z = 0

            # Usamos solve_ivp: Buscamos solución entre 0 y L
            if z_locs:  # Si le pasamos un valor de z_locs: El array de salida tendra z_locs elementos
                t_eval = np.linspace(0, fib.L, z_locs)
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
            else:
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)
    else:  # Sin la barra de progreso
        if z_locs:  # Si le pasamos un valor de z_locs: El array de salida tendra z_locs elementos
            t_eval = np.linspace(0, fib.L, z_locs)
            sol = solve_ivp(f_B, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
        else:
            sol = solve_ivp(f_B, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)
    

    zlocs = sol["t"]  #Puntos de z donde tenemos B(w,z)
    ysol  = sol["y"]  #Array, en cada elemento se tiene un subarray [B(w0,z0), B(w0,z1), ..., B(w0,zf)]
    

    zlocs = sol["t"]  #Puntos de z donde tenemos B(w,z)
    ysol  = sol["y"]  #Array, en cada elemento se tiene un subarray [B(w0,z0), B(w0,z1), ..., B(w0,zf)]
    print(sol["message"])

    ysol  = np.array(ysol)

    print(np.shape(ysol))
    
    ysol_transposed = ysol.T
    A_w = ysol_transposed
    for j in range( len(zlocs) ):
        A_w[j,:] = A_w[j,:] * np.exp(D_w * zlocs[j])
        # print("z = ", zlocs[j])
    #A_t = np.array([ttb.IFTopt(a_w) for a_w in A_w], dtype=complex)

    return zlocs, A_w, #A_t #Nos devuelve: zlocs = Posiciones donde calculamos la solución, A_w = Matriz con la evolución del espectro, A_t = Matriz con la evolución del pulso
