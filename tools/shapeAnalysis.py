#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to call in the PCA analysis of worm shapes

"""
import numpy as np
from sklearn import decomposition
from sklearn.decomposition import  pca
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
from .wormParam import *
import scipy.ndimage
import os
#################
"""
Helper Function
"""
def lowpass(sig,freq):
    fft_filt =np.ones(sig.size)
    fft_filt[freq:-freq]=0
    return np.fft.ifft(np.fft.fft(sig*fft_filt))

#################



def defineStates(theta,light,last_on,last_off,wormNumber=False,tp=1,lowpass_filter=False,com=False,smooth=0, ref_shape=None):
    #appends shapes at adjacent timepoints to make state vector
    #container
    Theta=np.zeros((len(theta),len(theta[0])))
    for i,val in enumerate(theta):
        if lowpass_filter:
            val=lowpass(val,lowpass_filter)
        Theta[i,:]=val
    #append
    P=np.zeros((len(theta)-tp,len(theta[0])*tp))
    light_plot=np.zeros(P.shape[0])
    t_on=np.zeros(P.shape[0])
    t_off=np.zeros(P.shape[0])
    WN=np.zeros(P.shape[0])
    COM=np.zeros((P.shape[0],2))
    if com:
        com=np.array(com)
    for i in range(0,P.shape[0]):
#        P[i,:]=scipy.ndimage.gaussian_filter(np.reshape(Theta[i:i+tp,:],P.shape[1]),smooth,mode='wrap')
        P[i,:]=np.reshape(Theta[i:i+tp,:],P.shape[1])
        light_plot[i]=np.mean(light[i:i+tp])
        t_on[i]=np.mean(last_on[i:i+tp])
        t_off[i]=np.mean(last_off[i:i+tp])
        if wormNumber:
            WN[i]=np.mean(wormNumber[i:i+tp])
        if type(com) == np.ndarray:
            COM[i,:]=np.mean(com[i:i+tp,:],0)
        
    #remove nan's
    flat=np.sum(P,1)
    ind_real=np.isfinite(flat)
    P=P[ind_real,:]
    light_plot=light_plot[ind_real]
    t_on=t_on[ind_real]
    t_off=t_off[ind_real]
    # L1 normalize radial measurements (ADDED 02.24.20)
    P = preprocessing.normalize(P, norm='l1', axis=1)
    if wormNumber:
        WN=WN[ind_real]
        if type(com) == np.ndarray:
            COM=COM[ind_real,:]
            return P,light_plot,t_on,t_off,WN, COM
        return P,light_plot,t_on,t_off,WN
    if type(com) == np.ndarray:
        COM=COM[ind_real,:]
        return P,light_plot,t_on,t_off,COM
    return P,light_plot,t_on,t_off

def pcaSpace(P,ncoord=20,scale=False,verbose=False,reference=False):
    #defines pca space for states and transforms into it
    #if verbose,return eigen vectors and values
    X=P#.copy()
    if scale:
        X=preprocessing.scale(X)
    if reference:
        if verbose:
            R,expVar,eig_val,eig_vector=referencePCA(reference, ncoord=ncoord, scale=scale,verbose=True)
            X=R.transform(X)
            return X,expVar, eig_val,eig_vector
        else:
            R=referencePCA(reference, ncoord=ncoord, scale=scale,verbose=True)
            X=R.transform(X)
            return X
    else: #makes PCA based on just this folder
        #make pca
        pca = decomposition.PCA(n_components=ncoord)        
        X = pca.fit_transform(X)
        #verbose
        if verbose:
            eig_val,eig_vector=np.linalg.eig(np.cov(P.T))
            return X,pca.explained_variance_ratio_, eig_val,eig_vector
        #else
        return X,pca.explained_variance_ratio_

def referencePCA(folder,ncoord=20,scale=False,verbose=False,tp=1,limFrame=False, retScaling=False, ref_mean=False):
    #defines reference pca space for states and returns functions
    #if verbose,return eigen vectors and values
    
    #check if reference PCA and scaling already defined and saved in folder if so, load and return
    files = os.listdir(folder)
    if 'PCA.pickle' in files:
        with open(folder+'PCA.pickle','rb') as f:
            pca = pickle.load(f)
        if scale:
            with open(folder+'SCALING.pickle','rb') as f:
                SCALING = pickle.load(f)
        if ref_mean:
            mean = np.load(folder+'REF_MEAN.npy')
        
    else:
        theta,light,last_on,last_off,wormNumber=loadAllWorms(folder,limFrame=limFrame)
        X,light,t_on,t_off,wormNumber=defineStates(theta,light,last_on,last_off,wormNumber,tp=tp)
        if ref_mean:
            mean=X.mean(axis=0)
            np.save(folder+'REF_MEAN.npy',mean)
        if scale:
            #remove mean
            X=preprocessing.scale(X, with_std=False)
            #make object for scaling variance
            SCALING=preprocessing.StandardScaler(with_mean=False).fit(X)
            X = SCALING.transform(X)
            with open(folder+'SCALING.pickle','wb') as f:
                pickle.dump(SCALING, f)
        #make pca
        pca = decomposition.PCA(n_components=ncoord,random_state=0)
        pca.fit(X)
        #save fit transformations:
        with open(folder+'PCA.pickle','wb') as f:
            pickle.dump(pca, f)
        
            
    
    #Return things
    if ref_mean:
        if verbose:
    #        eig_val,eig_vector=np.linalg.eig(np.cov(X.T))
            eig_val = pca.explained_variance_
            eig_vector = pca.components_
            if retScaling:
                return pca,pca.explained_variance_ratio_, eig_val,eig_vector,SCALING,mean
            else:
                return pca,pca.explained_variance_ratio_, eig_val,eig_vector,mean
        #else
        if retScaling:
            return pca,SCALING,mean
        return pca,mean
    else:
        if verbose:
    #        eig_val,eig_vector=np.linalg.eig(np.cov(X.T))
            eig_val = pca.explained_variance_
            eig_vector = pca.components_
            if retScaling:
                return pca,pca.explained_variance_ratio_, eig_val,eig_vector,SCALING
            else:
                return pca,pca.explained_variance_ratio_, eig_val,eig_vector
        #else
        if retScaling:
            return pca,SCALING
        return pca

def loadAndReferencePCA(folder,reference,scale=False,verbose=False,ncoord=20,tp=1,com=False, smooth=0):
    #define the reference PCA transform
    if verbose:
        R,expVar,eig_val,eig_vector=referencePCA(reference, ncoord=ncoord, scale=scale,verbose=True,tp=tp)
    else:
        R=referencePCA(reference, ncoord=ncoord, scale=scale,verbose=False,tp=tp)
    #containers
    X_all=np.array([])
    COM_all=np.array([])
    light=np.array([])
    last_on=np.array([])
    last_off=np.array([])
    wormNumber=np.array([])
    #individually transform each worm
    for i,filename in enumerate(sorted(os.listdir(folder))):
        if (not (filename[-6:]=='pickle')): #edit 083120: skip folder
            i-=1
            continue
        #load worm
        print(filename)
        worm=loadResults(folder+filename)
        if len(worm)>4: #handles if worm with com
            com=worm[-1]
            worm=worm[:-1]
            print(type(com),type(com[0]))
        #put result in right format
        if com:
            P,light_plot,t_on,t_off,COM=defineStates(*worm,com=com,tp=tp,smooth=smooth)
        else:
            P,light_plot,t_on,t_off=defineStates(*worm,com=com,tp=tp,smooth=smooth)
        #if no data for some reason,skip
        if len(P)==0:
            print('no Data for worm: ', filename)
            continue
        #scale if appropriate
        X=P.copy()
        if scale:
            X=preprocessing.scale(X)
        #transform to reference pca
        X=R.transform(X)
        #append reults
        if X_all.size==0:
            X_all=X.copy()
        else:
            X_all=np.append(X_all,X.copy(),0)
        
        if com:
            if COM_all.size==0:
                COM_all=COM.copy()
            else:
                COM_all=np.append(COM_all,COM.copy(),0)
        light=np.append(light,light_plot)
        last_on=np.append(last_on,t_on)
        last_off=np.append(last_off,t_off)
        wormNumber=np.append(wormNumber,np.ones(len(t_on))*i)
    #return things
    if com:
        if verbose:
            return X_all,light,last_on,last_off,wormNumber,COM_all,expVar,eig_val,eig_vector
        return X_all,light,last_on,last_off,wormNumber,COM_all
    if verbose:
        return X_all,light,last_on,last_off,wormNumber,expVar,eig_val,eig_vector
    return X_all,light,last_on,last_off,wormNumber

def loadAndReferencePCA_refScale(folder,reference,scale=False,verbose=False,
                                 ncoord=20,tp=1,com=False, smooth=0,):
    #define the reference PCA transform
    #Use this when abnormal variance but constant mean -->RNAi
    if verbose:
        R,expVar,eig_val,eig_vector, SCALING=referencePCA(reference, ncoord=ncoord, scale=scale,verbose=True,tp=tp,retScaling=True)
    else:
        R, SCALING=referencePCA(reference, ncoord=ncoord, scale=scale,verbose=False,tp=tp, retScaling=True)
    #containers
    X_all=np.array([])
    COM_all=np.array([])
    light=np.array([])
    last_on=np.array([])
    last_off=np.array([])
    wormNumber=np.array([])
    #individually transform each worm
    for i,filename in enumerate(sorted(os.listdir(folder))):
        if (not (filename[-6:]=='pickle')): #edit 083120: skip folder
            continue
        #load worm
        print(filename)
        worm=loadResults(folder+filename)
        if len(worm)>4: #handles if worm with com
            com=worm[-1]
            worm=worm[:-1]
            print(type(com),type(com[0]))
        #put result in right format
        if com:
            P,light_plot,t_on,t_off,COM=defineStates(*worm,com=com,tp=tp,smooth=smooth)
        else:
            P,light_plot,t_on,t_off=defineStates(*worm,com=com,tp=tp,smooth=smooth)
        #if no data for some reason,skip
        if len(P)==0:
            print('no Data for worm: ', filename)
            continue
        #scale if appropriate
        X=P.copy()
        Xmean = np.mean(X, axis=0)
        X = X-Xmean
        if scale:
            X=SCALING.transform(X) #Scale data according to reference PCA dataset
        #transform to reference pca
        X=R.transform(X)
        #append reults
        if X_all.size==0:
            X_all=X.copy()
        else:
            X_all=np.append(X_all,X.copy(),0)
        
        if com:
            if COM_all.size==0:
                COM_all=COM.copy()
            else:
                COM_all=np.append(COM_all,COM.copy(),0)
        light=np.append(light,light_plot)
        last_on=np.append(last_on,t_on)
        last_off=np.append(last_off,t_off)
        wormNumber=np.append(wormNumber,np.ones(len(t_on))*i)
    wormNumber=wormNumber-wormNumber.min() #edit 083120: resolves extra count for skipped forders
    #return things
    if com:
        if verbose:
            return X_all,light,last_on,last_off,wormNumber,COM_all,expVar,eig_val,eig_vector
        return X_all,light,last_on,last_off,wormNumber,COM_all
    if verbose:
        return X_all,light,last_on,last_off,wormNumber,expVar,eig_val,eig_vector
    return X_all,light,last_on,last_off,wormNumber

def loadAndReferencePCA_keepMean(folder,reference,scale=False,verbose=False,ncoord=20,tp=1,limFrame=False):
    #define the reference PCA transform
    if verbose:
        R,expVar,eig_val,eig_vector=referencePCA(reference, ncoord=ncoord, scale=scale,verbose=True,tp=tp,limFrame=limFrame)
    else:
        R=referencePCA(reference, ncoord=ncoord, scale=scale,verbose=False,tp=tp,limFrame=limFrame)
    #containers
    X_all=np.array([])
    light=np.array([])
    last_on=np.array([])
    last_off=np.array([])
    wormNumber=np.array([])
    #individually transform each worm
    for i,filename in enumerate(os.listdir(folder)):
        #load worm 
        worm=loadResults(folder+filename)
        #put result in right format
        P,light_plot,t_on,t_off=defineStates(*worm,tp=tp)
        #scale if appropriate
        X=P.copy()
        if scale:
            X=preprocessing.scale(X)
        #transform to reference pca moved outside of loop
#        X=R.transform(X)
        #append reults
        if X_all.size==0:
            X_all=X.copy()
        else:
            X_all=np.append(X_all,X.copy(),0)
        light=np.append(light,light_plot)
        last_on=np.append(last_on,t_on)
        last_off=np.append(last_off,t_off)
        wormNumber=np.append(wormNumber,np.ones(len(t_on))*i)
    #transform to reference pca
    X_all=R.transform(X_all)
    #return things
    if verbose:
        return X_all,light,last_on,last_off,wormNumber,expVar,eig_val,eig_vector
    return X_all,light,last_on,last_off,wormNumber

def loadAndReferencePCA_regenerating(folder,reference,sigma=20,scale=False,verbose=False,ncoord=20,tp=1,com=False, smooth=0):
    #define the reference PCA transform
    #remove mean as a gaussian average around point
    if verbose:
        R,expVar,eig_val,eig_vector, SCALING=referencePCA(reference, ncoord=ncoord, scale=scale,verbose=True,tp=tp,retScaling=True)
    else:
        R, SCALING=referencePCA(reference, ncoord=ncoord, scale=scale,verbose=False,tp=tp, retScaling=True)
    #containers
    X_all=np.array([])
    COM_all=np.array([])
    light=np.array([])
    last_on=np.array([])
    last_off=np.array([])
    wormNumber=np.array([])
    #individually transform each worm
    for i,filename in enumerate(sorted(os.listdir(folder))):
        #load worm
        print(filename)
        worm=loadResults(folder+filename)
        if len(worm)>4: #handles if worm with com
            com=worm[-1]
            worm=worm[:-1]
            print(type(com),type(com[0]))
        #put result in right format
        if com:
            P,light_plot,t_on,t_off,COM=defineStates(*worm,com=com,tp=tp,smooth=smooth)
        else:
            P,light_plot,t_on,t_off=defineStates(*worm,com=com,tp=tp,smooth=smooth)
        #if no data for some reason,skip
        if len(P)==0:
            print('no Data for worm: ', filename)
            continue
        #scale if appropriate
        X=P.copy()
        #define and subtract the localized mean
        localMean = scipy.ndimage.gaussian_filter1d(X, sigma=sigma, axis=0,truncate=2)
        X = X-localMean
        if scale:
            X=SCALING.transform(X) #Scale data according to reference PCA dataset
        #transform to reference pca
        X=R.transform(X)
        #append reults
        if X_all.size==0:
            X_all=X.copy()
        else:
            X_all=np.append(X_all,X.copy(),0)
        
        if com:
            if COM_all.size==0:
                COM_all=COM.copy()
            else:
                COM_all=np.append(COM_all,COM.copy(),0)
        light=np.append(light,light_plot)
        last_on=np.append(last_on,t_on)
        last_off=np.append(last_off,t_off)
        wormNumber=np.append(wormNumber,np.ones(len(t_on))*i)
    #return things
    if com:
        if verbose:
            return X_all,light,last_on,last_off,wormNumber,COM_all,expVar,eig_val,eig_vector
        return X_all,light,last_on,last_off,wormNumber,COM_all
    if verbose:
        return X_all,light,last_on,last_off,wormNumber,expVar,eig_val,eig_vector
    return X_all,light,last_on,last_off,wormNumber

def loadAndReferencePCA_100920(folder,reference,scale=False,verbose=False,
                                 ncoord=20,tp=1,com=False, smooth=0,):
    #define the reference PCA transform
    #Use this when abnormal variance but constant mean -->RNAi
    if verbose:
        R,expVar,eig_val,eig_vector, SCALING, mean=referencePCA(reference, ncoord=ncoord,
                                                                scale=scale,verbose=True,tp=tp,
                                                                retScaling=True, ref_mean=True)
    else:
        R, SCALING, mean=referencePCA(reference, ncoord=ncoord, scale=scale,
                                      verbose=False,tp=tp, retScaling=True, ref_mean=True)
    #containers
    X_all=np.array([])
    COM_all=np.array([])
    light=np.array([])
    last_on=np.array([])
    last_off=np.array([])
    wormNumber=np.array([])
    #individually transform each worm
#    plt.plot(mean)
    for i,filename in enumerate(sorted(os.listdir(folder))):
        if (not (filename[-6:]=='pickle')): #edit 083120: skip folder
            continue
        #load worm
        print(filename)
        worm=loadResults(folder+filename)
        if len(worm)>4: #handles if worm with com
            com=worm[-1]
            worm=worm[:-1]
            print(type(com),type(com[0]))
        #put result in right format
        if com:
            P,light_plot,t_on,t_off,COM=defineStates(*worm,com=com,tp=tp,smooth=smooth)
        else:
            P,light_plot,t_on,t_off=defineStates(*worm,com=com,tp=tp,smooth=smooth)
        #if no data for some reason,skip
        if len(P)==0:
            print('no Data for worm: ', filename)
            continue
        #scale if appropriate
        X=P.copy()
        Xmean = np.mean(X, axis=0)
        X = X*(mean/Xmean)
        Xmean2 = np.mean(X, axis=0)
        X=X-mean
        plt.plot(Xmean,c='grey',alpha=.5)
        plt.plot(Xmean2,c='r',alpha=.5)
        plt.show()
        SCALING=preprocessing.StandardScaler(with_mean=False).fit(X)
        X = SCALING.transform(X)
#        X = X-mean
#        if scale:
#            X=SCALING.transform(X) #Scale data according to reference PCA dataset
        #transform to reference pca
        X=R.transform(X)
        #append reults
        if X_all.size==0:
            X_all=X.copy()
        else:
            X_all=np.append(X_all,X.copy(),0)
        
        if com:
            if COM_all.size==0:
                COM_all=COM.copy()
            else:
                COM_all=np.append(COM_all,COM.copy(),0)
        light=np.append(light,light_plot)
        last_on=np.append(last_on,t_on)
        last_off=np.append(last_off,t_off)
        wormNumber=np.append(wormNumber,np.ones(len(t_on))*i)
    wormNumber=wormNumber-wormNumber.min() #edit 083120: resolves extra count for skipped forders
    #return things
    if com:
        if verbose:
            return X_all,light,last_on,last_off,wormNumber,COM_all,expVar,eig_val,eig_vector
        return X_all,light,last_on,last_off,wormNumber,COM_all
    if verbose:
        return X_all,light,last_on,last_off,wormNumber,expVar,eig_val,eig_vector
    return X_all,light,last_on,last_off,wormNumber
###############################################3
##HELPER functions
def binAvg(x,y,bi=np.linspace(0,1,10)):
    n1,b,p=plt.hist(x,bins=bi)
    n2,b,p=plt.hist(x,bins=bi,weights=y)
    plt.close('all')
    return n2/n1