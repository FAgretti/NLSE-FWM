import numpy as np
import cupy as cp
from numpy.fft import fftshift

class Sim:
    def __init__(self, puntos, Tmax):
        self.puntos = puntos                         #Número de puntos sobre el cual tomar el tiempo
        self.Tmax   = Tmax
        self.paso_t = 2.0*Tmax/puntos
        self.tiempo = cp.arange(-puntos/2,puntos/2)*self.paso_t
        self.dW     = cp.pi/Tmax
        self.freq   = cp.fft.fftshift( cp.pi * cp.arange(-puntos/2,puntos/2) / Tmax )/(2*cp.pi)
        self.fs     = 1/self.paso_t

class Fibra:
    def __init__(self, L, gamma, gamma1, alpha, lambda0, betas):
        self.L  = L         #Longitud de la fibra
        self.betas = betas  #Vector con los coeficientes beta
        self.gamma = gamma  #gamma de la fibra, para calcular SPM
        self.gamma1= gamma1 #gamma1 de la fibra, para self-steepening
        self.alpha = alpha  #alpha de la fibra, atenuación
        self.lambda0 = lambda0 #Longitud de onda central
        self.omega0  = 2*cp.pi* 299792458 * (1e9)/(1e12) /lambda0 #Frecuencia (angular) central
    
    #---Algunos métodos útiles---
    #Método para pasar de omega a lambda
    def omega_to_lambda(self, w):  #Función para pasar de Omega a lambda.
        return 2*cp.pi* 299792458 * (1e9)/(1e12)/(self.omega0+w)
    def lambda_to_omega(self,lam): #Función para pasar de lambda a Omega.
        return 2*cp.pi*299792458 * (1e9)/(1e12) * (1/lam - 1/self.lambda0)
    #Método para calcular gamma en función de omega
    def gamma_w(self, w, wavelength=False):
        if wavelength:
            w = self.lambda_to_omega(w)
        return self.gamma + self.gamma1 * w
    #Método para calcular beta2 en función de omega
    def beta2_w(self, w, wavelength=False):
        if wavelength:
            w = self.lambda_to_omega(w)
        if self.betas != 0:
            beta2 = 0
            for i, beta in enumerate(self.betas):
                beta2 += beta * w**i / cp.math.factorial(i)
        else:
            beta2 = self.beta2 + self.beta3 * w
        return beta2