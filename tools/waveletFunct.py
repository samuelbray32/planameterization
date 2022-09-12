#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:28:50 2019

@author: sam
"""
import numpy as np
import scipy.signal
import pywt

def haar_wavelet(X,a,mode='same'):
    if a==0:
        return X
    haar = np.ones(2*a)
    haar[a:] = -1
    return np.convolve(X,haar,mode=mode)/a#[a:-a+1]
    
def haar_wavelet_com(COM,a,mode='same'):
    haar = np.ones(2*a)
    haar[:a] = -1
    c_x=np.convolve(COM[:,0],haar,mode=mode)/2/a#[a:-a+1]
    c_y=np.convolve(COM[:,1],haar,mode=mode)/2/a#[a:-a+1]
    return (c_x**2+c_y**2)**.5

def sin_wavelet(X,a,normalize=False):
    x = np.linspace(0,-2*np.pi,a)
    sin_wav = np.cos(x)
    if normalize:
        return np.convolve(X,sin_wav,mode='valid')/a
    return np.convolve(X,sin_wav,mode='valid')

def sin_wavelet_2(X,a,normalize=False):
    x = np.linspace(0,5*-2*np.pi,a)
    sin_wav = np.sin(x)
    x2 = np.linspace(0,1,a)
    gaus = np.exp(-(x2-.5)**2/.2**2)
    sin_wav *= gaus    
    if normalize:
        sin_wav = sin_wav/a
    return np.convolve(X,sin_wav,mode='valid')


def ricker_wavelet(X,a):
    wav = scipy.signal.ricker(a,a//10)
    return np.convolve(X,wav,mode='valid')

def morlet_wavelet(X,a):
    wav = np.real(scipy.signal.morlet(2*a)[a:]/a)
    return np.convolve(X,wav,mode='valid')

def db2_wavelet(X,a,ret_scale=False):
    wav = pywt.Wavelet('db2').wavefun(level=a)[1]
    if ret_scale:
        return np.convolve(X,wav,mode='valid'), wav.size
    return np.convolve(X,wav,mode='valid')

def gauss_smooth(X,a,w):
    wav=scipy.signal.gaussian(w,a)
    return np.convolve(X,wav/np.sum(wav),mode='same')

## In[]
#a=100
#x = np.linspace(0,5*-2*np.pi,a)
#sin_wav = np.cos(x)
#x2 = np.linspace(0,1,a)
#gaus = np.exp(-(x2)/.3)
#
#plt.plot(sin_wav)
#plt.plot(gaus)
#plt.plot(sin_wav*gaus) 
#
#print(np.sum(sin_wav*gaus))
#
#
## In[]
#z = scipy.signal.morlet(20)
#plt.plot(z[z.size//2:])




