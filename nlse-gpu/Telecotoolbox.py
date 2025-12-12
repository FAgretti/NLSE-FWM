import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.signal as signal
import scipy.fftpack as fftpack


def cajon(VarIn,a=-0.5,b=0.5):
    return(VarIn>=a)*(VarIn<=b)

def tfourier(varIn,fs):
    # Se calcula la fft de la entrada
    varIn_tdf = np.fft.ifft(varIn)
    varIn_tdf = np.fft.ifftshift(varIn_tdf)

    # Se define el vector de frecuencias
    sampNum = varIn.size
    if np.remainder(sampNum,2) == 0:
        k = np.arange(-sampNum/2,sampNum/2,1)
    else:
        k = np.arange(-(sampNum-1)/2-1,(sampNum-1)/2,1)    
    fk = k*fs/sampNum

    # Se escala el espectro
    varOut = varIn_tdf*sampNum/fs
    return varOut,fk

def itfourier(varIn,fs,sampnum):
    return np.fft.fft(np.fft.fftshift(varIn*fs))/sampnum

@cp.fuse()
def tfourierGPU(varIn,fs):
    # Se calcula la fft de la entrada
    varIn_tdf = cp.fft.ifft(varIn)
    varIn_tdf = cp.fft.ifftshift(varIn_tdf)

    # Se define el vector de frecuencias
    sampNum = varIn.size
    if cp.remainder(sampNum,2) == 0:
        k = cp.arange(-sampNum/2,sampNum/2,1)
    else:
        k = cp.arange(-(sampNum-1)/2-1,(sampNum-1)/2,1)    
    fk = k*fs/sampNum

    # Se escala el espectro
    varOut = varIn_tdf*sampNum/fs
    return varOut,fk

def itfourierGPU(varIn,fs,sampnum):
    return cp.fft.fft(cp.fft.fftshift(varIn*fs))/sampnum

def t_a_freq(t_o_freq):
    return fftpack.fftfreq( len(t_o_freq) , d = t_o_freq[1] - t_o_freq[0])

def gauss(x,mu,sigma,p0):
    return np.sqrt(p0)*np.exp(-0.5*((x-mu)/sigma)**2)

def gaussGPU(x,mu,sigma,p0):
    return cp.sqrt(p0)*cp.exp(-0.5*((x-mu)/sigma)**2)

def supergaussGPU(x,mu,sigma,p0,m):
    return cp.sqrt(p0)*cp.exp(-0.5*((x-mu)/sigma)**2*m)

def hypsec(x,a,b,p0):
    return np.sqrt(p0)/(np.cosh(a*x+b))

def hypsecGPU(x,a,b,p0):
    return cp.sqrt(p0)/(cp.cosh(a*x+b))

def Gpulse(Pot, ancho, chirp,t):
    """Pot = potencia, ancho = ancho del pulso, chirp = chirp del pulso, t = vector de tiempo"""
    if chirp:
        return np.sqrt(Pot)*np.exp(-0.5*((t-ancho/2)/chirp)**2)
    else:
        return np.sqrt(Pot)*np.exp(-0.5*(t/ancho)**2)

def PRBS(tasa,t):
    """tasa = tasa de bits, t = vector de tiempo"""
    numOcurrencias = int((t[-1]-t[0])*tasa)
    print(numOcurrencias)
    seq = np.random.randint(0,2,numOcurrencias)
    seq = np.repeat(seq, t.size//numOcurrencias)
    if seq.size<t.size:
        seq = np.append(seq,np.random.randint(0,2)*np.ones(t.size-seq.size))
    return seq

def Butterworth(señal,frec,fs,orden):
    """frec = frecuencia de corte, fs = frecuencia de muestreo, orden = orden del filtro"""
    frecNorm = frec/(fs/2)
    b,a = signal.butter(orden,frecNorm,btype='low')
    return signal.filtfilt(b,a,señal)

def TFopt(pulso):
    if pulso.ndim == 1:
        return fftpack.ifft(pulso) * len(pulso)
    elif pulso.ndim == 2:
        return fftpack.ifft(pulso, axis=1) * pulso.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

def IFTopt(espectro):
    if espectro.ndim == 1:
        return fftpack.fft(espectro) / len(espectro)
    elif espectro.ndim == 2:
        return fftpack.fft(espectro, axis=1) / espectro.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")


def TFoptGPU(pulso):
    if pulso.ndim == 1:
        return cp.fft.ifft(pulso) * len(pulso)
    elif pulso.ndim == 2:
        return cp.fft.ifft(pulso, axis=1) * pulso.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

def IFToptGPU(espectro):
    if espectro.ndim == 1:
        return cp.fft.fft(espectro) / len(espectro)
    elif espectro.ndim == 2:
        return cp.fft.fft(espectro, axis=1) / espectro.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")
