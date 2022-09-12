#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:15:25 2019

@author: gary
"""

import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
import skvideo.datasets
from skimage.morphology import watershed
from skimage.morphology import skeletonize,medial_axis
from scipy import ndimage as ndi
from skimage import filters, measure
import time
#In[]
sz=3000
nPoints=500

def radialShape(frame,nPoints=20,sz=1000,ret_com=False, select_largest=False):
    #find worm object
    lab=measure.label(frame)
    regions=measure.regionprops(lab)
    img=np.array([])
    global_com=(np.nan,np.nan)
    if select_largest: #added 05.06.20  use for the planParty to discriminate most likely to be your worm
        max_s=0
        for i,props in enumerate(regions):
            if (props.convex_area<sz) or (props.convex_area>80000):  # or (props.eccentricity>.99)
                continue
            if props.convex_area>max_s:
                max_s=props.convex_area
                img=np.pad(props.filled_image,(10,10),'constant',constant_values=(0,0))
                global_com=props.centroid
    else:
        min_d = np.inf #added 02.25.20 used to select valid object closest to center of region. Helpful for roiFree tracking
        c = np.array([frame.shape[0]/2,frame.shape[1]/2])
        for i,props in enumerate(regions):
            if (props.convex_area<sz) or (props.convex_area>80000):  # or (props.eccentricity>.99)
                continue
            if np.linalg.norm(np.array(props.centroid[0],props.centroid[1])-c)<min_d:
                min_d = np.linalg.norm(props.centroid-c)
                img=np.pad(props.filled_image,(10,10),'constant',constant_values=(0,0))
                global_com=props.centroid
    if img.shape[0]==0:
        print('no good object')
        if ret_com:
            return np.ones(nPoints)*np.nan,global_com
        return np.ones(nPoints)*np.nan
    #smooth edges
    img=filters.gaussian(img,2)>.1 #sigma and threshold chosen by trial and error
#    img[(img<.1)]=0
#    img[(img>0)]=1
    #define perimeter
    perim=measure.find_contours(img.astype('int'),.5)[0]
    #define com
    loc=np.where(img>0)
    com=[np.mean(loc[0]),np.mean(loc[1])]
    #select points along com to evaluate distance
    """ColoringVersion
    #pos=np.linspace(0,len(perim),nPoints+1).astype(int)
    #pos=pos[:-1]
    #metric=[]
    #clr=plt.cm.coolwarm(np.linspace(0.1,0.9,nPoints+3))
    #for q,i in enumerate(pos):
    #    metric.append(np.linalg.norm(com-perim[i]))
    #    plt.plot([com[1],perim[i][1]],[com[0],perim[i][0]],zorder=10,c=clr[q])#c='cyan')
    #metric=metric/np.mean(metric)
    """
    pos=np.linspace(0,len(perim),nPoints+1).astype(int)
    pos=pos[:-1]
    metric=[]
    for i in pos:
        metric.append(np.linalg.norm(com-perim[i]))
   # metric=metric/np.mean(metric) #cut 05.05.20, L1 normed anyway so keep this information
    #Align radial positions -->for now based on max value
    loc=np.where(metric==np.max(metric))
    #in case multiple at max value
    if len(loc[0])>1:
        loc=loc[0]
    metric=np.roll(metric,-loc[0])
    if ret_com:
        return metric, global_com
    return metric


