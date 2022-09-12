#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:49:07 2020

@author: sam
"""

import skvideo
from radialShape import *
from scipy.interpolate import splprep, splev
import pickle
import os 
from skimage.filters import threshold_otsu, threshold_yen
import time
from tqdm import tqdm
from wormParam import *

def extractAllWorms_ROIfree(vidName,folder,COM_init,roi_size = 500, ang_len=100,
                            LD_thresh=60, size_lim=[2000,2000],binar=False,
                            rng_lo=False,rng_hi=False,save=10**4,com=False,
                            intense_region=False,background=False,thresh_scale=.95,
                            binarize_percentile=99.9,invert=False):

    print('it\'s the new one!')
    print('com', com)
    #set defaults if necessary
    if not rng_lo:
        rng_lo=[0,LD_thresh]
    if not rng_hi:
        rng_hi=[LD_thresh,255]
    n=len(COM_init)
    #Iterate through frames to extract shape params
    #per worm containers
    theta = [[] for x in range(n)] #states
    COM = [[] for x in range(n)] #com tracking
    bi_thresh = np.zeros(n)
    #overall state containers
    light = [] #current light state
    intense_all = []#track mean intensity
    last_hi = [] #times since last hi and lo state
    last_lo = []
    lh=0 #ints for tracking previous light switches
    ll=0
    
    videogen = skvideo.io.vreader(vidName)
    i=0
    a=time.time()
    for frame in tqdm(videogen,leave=True, position=0):
        
        #080922
        # if invert:
            # frame=-frame+255
        #skip initial frames
        if i<5:
            i+=1
            #set intense region as whole frame if not passed
            if intense_region==False:
                intense_region=((0,frame.shape[0]),(0,frame.shape[1]))
            continue
        #track progress
#        if i%100==0:
#    #        print(i)
#            print(i,time.time()-a)
#            a=time.time()
        i+=1
        # print('check1',i)
#        b=time.time()
        #handle global light condition tracking
        intense=np.mean(frame[intense_region[0][0]:intense_region[0][1],intense_region[1][0]:intense_region[1][1],0])
        intense_all.append(intense)
        #for light time tracking--records for every frame regardless of if good segment
        if intense>LD_thresh:
            lh=i
        else:
            ll=i
        last_hi.append(i-lh)
        last_lo.append(i-ll)   
        # print('check2',i)
        #define if light or dark frame
        #hi
        if (intense>rng_hi[0]) and (intense<rng_hi[1]):
            light.append(1)
        #lo
        elif (intense>rng_lo[0]) and (intense<rng_lo[1]):
            light.append(0)
        #neither--adjustment frames, append nan's  
        else:
            light.append(np.nan)
            for t in theta: 
                t.append(np.ones(ang_len)*np.nan)
            continue
#        print('lightmanage',time.time()-b)
        # print('check3',i)
        if type(background)==np.ndarray:
            frame = np.array(frame).copy().astype(float)
            frame=(frame[:,:,0]-background)[:,:,None]  
        # print('pre-loop')
        #extract each worm
        for j in range (0,n):
            # print('check', j)
            #find last known com for this worm
            if len (COM[j])==0:
                last_com = COM_init[j]
            else:
                z=-1
                while np.abs(z)<len(COM[j]):
                    if np.isfinite(COM[j][z][0]) and np.isfinite(COM[j][z][1]):
                        break
                    z-=1
                if np.abs(z)==len(COM[j]):
                    last_com = COM_init[j]
                else:
                    last_com = COM[j][z]

            #define ROI around last known point
            ROI = ((int(max(last_com[0]-roi_size,0)),int(min(last_com[0]+roi_size,frame.shape[0]))),
                   (int(max(last_com[1]-roi_size,0)),int(min(last_com[1]+roi_size,frame.shape[1]))))
            #use for dubugging (yay!)
            # plt.imshow(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0])
            # plt.ginput(n=1, timeout=-1) 
            if frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0].size==0:
                print('ROI ERROR: ', j, last_com)
            #Update threshold if: not set, lighting has changed, or bad previous frame
            if invert:
                bi_thresh[j]=thresh_scale*np.percentile(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0],binarize_percentile)
                img=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0]<bi_thresh[j]
            else:
                bi_thresh[j]=thresh_scale*np.percentile(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0],binarize_percentile)
    #                bi_thresh[j]=thresh_scale*threshold_otsu(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0])
                #binarize
                img=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0]>bi_thresh[j]
            
            if com:
                temp=radialShapeFromFrame(img,ang_len,worm_sz=size_lim[light[-1]],com=com)
                theta[j].append(temp[0])
#                com_this = (temp[1][0]-roi_size+last_com[0], temp[1][1]-roi_size+last_com[1])
#                COM[j].append((temp[1][0]-roi_size+last_com[0], temp[1][1]-roi_size+last_com[1]))
#                COM[j].append((temp[1][0]-img.shape[0]/2+last_com[0], temp[1][1]-img.shape[1]/2+last_com[1])) # bug fix 02.27.20 --error with cropped ROI from edge of frame
                #bug fix 04.23.20 --if shape truncated by edge previous com NOT in center of frame. 
                new_com=[0,0]                
                if ROI[0][0]==0:
                    new_com[0]=temp[1][0]
                elif ROI[0][1]==frame.shape[0]:
                    new_com[0]=frame.shape[0]-img.shape[0]+temp[1][0]
                else:
                    new_com[0]=temp[1][0]-img.shape[0]/2+last_com[0]
                if ROI[1][0]==0:
                    new_com[1]=temp[1][1]
                elif ROI[1][1]==frame.shape[1]:
                    new_com[1]=frame.shape[1]-img.shape[1]+temp[1][1]
                else:
                    new_com[1]=temp[1][1]-img.shape[1]/2+last_com[1]   
                COM[j].append((new_com[0],new_com[1])) 
                # print(len(COM[j]))
                    
                    
                    
                    
                    
                    
            else:
                theta[j].append(radialShapeFromFrame(img,ang_len,worm_sz=size_lim[light[-1]],com=com))
            if i%save==0:
#                plt.plot(intense_all)
                name=str(folder+'reg'+str(j))
                if com:
                    saveResults(name,theta[j],np.array(light),last_lo,last_hi,COM[j])
                else:
                    saveResults(name,theta[j],np.array(light),last_lo,last_hi)
#            print('worm',j,time.time()-b)

    if com:
        for j in range (0,n):
            # print(j, COM[j])
            name=str(folder+'reg'+str(j))
            saveResults(name,theta[j],np.array(light),last_lo,last_hi,COM[j])
    else:
        for j in range (0,n):
            name=str(folder+'reg'+str(j))
            saveResults(name,theta[j],np.array(light),last_lo,last_hi)
    plt.plot(intense_all)
    # return last COM to pass to next video (needed for multipart)
    com_end = [com[-1] for com in COM]
    return com_end