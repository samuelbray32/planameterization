#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:36:17 2019

@author: gary
"""

import numpy as np
import skvideo.io


def intensityTimeseries(vidName, nFrames, ROI, channel=0):
    intense=[]
    videogen = skvideo.io.vreader(vidName)
    for i, frame in enumerate(videogen):
        if i%1000==0:
            print(i)
        if i>nFrames:
            break
        intense.append(np.mean(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel]))
    return np.array(intense)

import imageio
def intensityTimeseries_2(vidName, nFrames, ROI, channel=0):
    intense=[]
    videogen = imageio.get_reader(vidName,  'ffmpeg')
    for i, frame in enumerate(videogen):
        if i%1000==0:
            print(i)
        if i>nFrames:
            break
        intense.append(np.mean(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel]))
    return np.array(intense)

#def pulseBackgrounds(vidName, nFrames, ROI, sample, thresh_hi,thresh_lo, thresh_pulse, channel=0):
#    #containers for backgrounds
#    back_hi=np.empty((0,0))
#    back_lo=np.empty((0,0))
#    
#    #go frame by frame through video
#    videogen = skvideo.io.vreader(vidName)
#    max_samp=np.max(sample)
#    for i,frame in enumerate(videogen):
#        if i%100==0:
#            print(i)
#        if i>max_samp:
#            break
#        if not(i in sample):
#            continue
#        #determine if frame assignment is to either of the hi or lo groups
#        intense=np.mean(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel])
#        if (intense>thresh_hi) or (intense<thresh_lo):
#            continue;
#        #if hi
#        if intense>thresh_pulse:
#            print('hi')
#            #if not yet assigned
#            if back_hi.shape[0]==0:
#                back_hi=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel].copy()
#            #compare for new minima    
#            back_hi=np.minimum(back_hi,frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel])
#        #same thing but if lo
#        else:
#            print('lo')
#            if back_lo.shape[0]==0:
#                back_lo=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel].copy()
#            back_lo=np.minimum(back_lo,frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel])
#    return back_lo,back_hi
#    

def pulseBackgrounds(vidName, nFrames, ROI, sample, rng_lo,rng_hi, channel=0):
    #containers for backgrounds
    back_hi=np.empty((0,0))
    back_lo=np.empty((0,0))
    
    #go frame by frame through video
    videogen = skvideo.io.vreader(vidName)
    max_samp=np.max(sample)
    for i,frame in enumerate(videogen):
        if i%100==0:
            print(i)
        if i>max_samp:
            break
        if not(i in sample):
            continue
        #determine if frame assignment is to either of the hi or lo groups
        intense=np.mean(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel])
        #if hi
        if (intense>rng_hi[0]) and (intense<rng_hi[1]):
            print('hi')
            #if not yet assigned
            if back_hi.shape[0]==0:
                back_hi=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel].copy()
            #compare for new minima    
            back_hi=np.minimum(back_hi,frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel])
        #same thing but if lo
        elif (intense>rng_lo[0]) and (intense<rng_lo[1]):
            print('lo')
            if back_lo.shape[0]==0:
                back_lo=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel].copy()
            back_lo=np.minimum(back_lo,frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel])
    return back_lo,back_hi

def constBackground(vidName, nFrames, ROI, filename=False, channel=0):
    #containers for backgrounds
    back=255*np.ones((ROI[0][1]-ROI[0][0],ROI[1][1]-ROI[1][0]))
    
    #go frame by frame through video
    videogen = skvideo.io.vreader(vidName)
    for i,frame in enumerate(videogen):
        if i%1000==0:
            if filename:
#                print('save')
                np.save(filename,back)
            print(i)
        if i>nFrames:
            break;
        back=np.minimum(back,frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],channel])
    return back
    
