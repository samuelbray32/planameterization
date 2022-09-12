#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:11:44 2019

@author: gary
"""


from radialShape import *
from scipy.interpolate import splprep, splev
import pickle
import os 
from skimage.filters import threshold_otsu, threshold_yen
import time

####################
#File managment
def saveResults(name,theta,light,last_on,last_off,com=False):
    Results=(theta,light,last_on,last_off)
    if (type(com)==np.ndarray) or com:
        Results=(theta,light,last_on,last_off,com)
    with open(str(name+'.pickle'), 'wb') as f:
        pickle.dump(Results, f)
    return

def loadResults(name):
#    with open(str(name+'.pickle'), "rb") as input_file:
#    print(name)
    with open(name, "rb") as input_file:
        e = pickle.load(input_file)
    return e #(theta,light,last_on,last_off)

def loadAllWorms(folder,limFrame=False,com=False):
    #Load all files in folder
    #define containers
    theta=[]
    light=[]
    last_on=[]
    last_off=[]
    wormNumber=[]
    COM = []
    if not limFrame:
        limFrame=-1
    
    for i,filename in enumerate(sorted(os.listdir(folder))):
        print(filename)
        R=loadResults(folder+filename)
    #         if limFrame:
    #             for r in R:
    #                 r=r[:limFrame]
         #squish this worm onto the others
        theta.extend(R[0][:limFrame])
        light.extend(R[1][:limFrame])
        last_on.extend(R[2][:limFrame])
        last_off.extend(R[3][:limFrame])
        wormNumber.extend(np.ndarray.tolist(np.ones(len(R[1][:limFrame]))*i))
        if com and (len(R)==5):
            COM.extend(R[4])
         #add a nan between worms to keep data separate
        theta.append(theta[-1]*np.nan)
        light.extend([np.nan])
        last_on.extend([0])
        last_off.extend([0])
        wormNumber.extend([-1])
#         COM.extend((np.nan,np.nan))
#         print(len(theta))
    if com:
       return theta,light,last_on,last_off,wormNumber,COM 
    return theta,light,last_on,last_off,wormNumber

def mergeWorms(folder1,folder2,destination):
    ##not efficient if stupid big numbers of worms in folder but ok in practice
    for i,filename1 in enumerate(os.listdir(folder1)):
        theta=[]
        light=[]
        last_on=[]
        last_off=[]
        wormNumber=[]
        com=[]
        R1=loadResults(folder1+filename1)
        theta.extend(R1[0])
        light.extend(R1[1])
        last_on.extend(R1[2])
        last_off.extend(R1[3])
        if len(R1)>4: #if com
            com.extend(R1[4])
        for j,filename2 in enumerate(os.listdir(folder2)):
            if not(filename1==filename2):
                continue
            print(filename1,filename2)
            R2=loadResults(folder2+filename2)
            theta.extend(R2[0])
            light.extend(R2[1])
            # append last value until reset
            last_on2=R2[2]
            i=0
            while i<len(last_on2) and last_on2[i]>0:
                last_on2[i] += last_on[-1]
                i += 1
            last_on.extend(last_on2)
            # append last value until reset
            last_off2=R2[3]
            i=0
            while i<len(last_off2) and last_off2[i]>0:
                last_off2[i] += last_off[-1]
                i += 1
            last_off.extend(last_off2)
            if len(R2)>4: #if com
                com.extend(R2[4])
                print(filename1[:-7])
                saveResults(destination+filename1[:-7],theta,light,last_on,last_off,com) 
            else:
                saveResults(destination+filename1[:-7],theta,light,last_on,last_off) 
    return


#########################
"""
drawROI--helper func to draw your ROI as you make them

extractAllWorms--pulls out radial shape of all worms identified in list of ROI (ROI_all)
--automated binarizing--no option
--rng_{hi,lo} determined just by LD_thresh unless told otherwise
"""    
def drawROI(ROI):
    plt.plot([ROI[1][0],ROI[1][0]],[ROI[0][0],ROI[0][1]],c='r')
    plt.plot([ROI[1][1],ROI[1][1]],[ROI[0][0],ROI[0][1]],c='r')
    plt.plot([ROI[1][0],ROI[1][1]],[ROI[0][0],ROI[0][0]],c='r')
    plt.plot([ROI[1][0],ROI[1][1]],[ROI[0][1],ROI[0][1]],c='r')
    return

def extractAllWorms(vidName,folder,ROI_all,ang_len=100, LD_thresh=60,
                    size_lim=[2000,2000],binar=False,rng_lo=False,rng_hi=False,
                    save=10**4,com=False,intense_region=False,background=False,
                    thresh_scale=.95):

    #set defaults if necessary
    if not rng_lo:
        rng_lo=[0,LD_thresh]
    if not rng_hi:
        rng_hi=[LD_thresh,255]
    n=len(ROI_all)
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
    for frame in videogen:
        #skip initial frames
        if i<5:
            i+=1
            #set intense region as whole frame if not passed
            if intense_region==False:
                intense_region=((0,frame.shape[0]),(0,frame.shape[1]))
            continue
        #track progress
        if i%100==0:
    #        print(i)
            print(i,time.time()-a)
            a=time.time()
        i+=1
        
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
        
        if background:
            frame=frame-background
        #extract each worm
        for j in range (0,n):
#            b=time.time()
            ROI=ROI_all[j]
            #Update threshold if: not set, lighting has changed, or bad previous frame
            if True:#(bi_thresh[j]==0) or (not(np.isfinite(theta[j][-1][0]))):# or (np.abs(intense_all[-1]-intense_all[-2])>.2) : 
    #            print('changed')
#                bi_thresh[j]=1.5*threshold_otsu(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0])
#                bi_thresh[j]=1*threshold_otsu(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0])
#                bi_thresh[j]=thresh_scale*np.max(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0])
                bi_thresh[j]=thresh_scale*np.percentile(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0],99.9)
            #binarize
            img=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0]>bi_thresh[j]
            if com:
                temp=radialShapeFromFrame(img,ang_len,worm_sz=size_lim[light[-1]],com=com)
                theta[j].append(temp[0])
                COM[j].append(temp[1])
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
        saveResults(name,theta[j],np.array(light),last_lo,last_hi,COM[j])
    else:
        saveResults(name,theta[j],np.array(light),last_lo,last_hi)
    plt.plot(intense_all)
    return




###########################
"""
#shape extraction
shapeFromFrame:  pulls out skeletonized shape parameterized as either angles of fft

radialShapeFromFrame:  New standard parameterization
    Based on Ingmar and Jochen 2014

"""
def shapeFromFrame(frame,pSize=10,worm_sz=1000,fft=False):
    ###high level function, takes in binarized frame and returns angle parameters
    #blur edges to smooth
    #median kernel
    kern=np.ones((3,3))
    kern[0,0]=0
    kern[0,2]=0
    kern[2,0]=0
    kern[2,2]=0
    manip=filters.median(frame,kern)
    #Gaussian and re-threshold
    manip=filters.gaussian(manip,3)
    manip[(manip<.1)]=0
    manip[(manip>0)]=1
    #if weird binarizing, skip
    if np.sum(manip)/manip.size>.15:
        print('binarize exception')
        return(np.ones(pSize)*np.nan)
    #make skeleton
    sk=prunedSkeleton(manip,sz=worm_sz,outline=False)
#    plt.imshow(sk)
    if np.sum(sk)<3:
        print('skeleton error')
        return(np.ones(pSize)*np.nan)
    #splineFit Skeleton
    spline=makeSpline(sk,pSize+2)
    #if too short a curve (disconnected graph)
    if not spline:
        print('discontinuous graph')
        return(np.ones(pSize)*np.nan)
    #if returning fft parametrization
    if fft:
        return(fftFromSpline(spline))
    #fit angle parameterization--if not fft
    return angleFromSpline(spline)

def radialShapeFromFrame(frame,pSize=10,worm_sz=1000,com=False,select_largest=False):
    ###high level function, takes in binarized frame and returns radial distances
    ###based on Ingmar and Jochen paper
    """
    #blur edges to smooth
    #median kernel
    kern=np.ones((3,3))
    kern[0,0]=0
    kern[0,2]=0
    kern[2,0]=0
    kern[2,2]=0
    manip=filters.median(frame,kern)
    #Gaussian and re-threshold
    manip=filters.gaussian(manip,3)
    manip[(manip<.1)]=0
    manip[(manip>0)]=1
    """
    #Gaussian and re-threshold
    frame=filters.gaussian(frame,3)
    frame=frame>.5
#    manip=frame.copy()/np.max(frame)
    #if weird binarizing, skip
#    if np.sum(frame)/frame.size>.15:
#        print('binarize exception')
#        return(np.ones(pSize)*np.nan)
    
    return radialShape(frame,pSize,worm_sz,ret_com=com,select_largest=select_largest)

def radialShapeFromFrame_Debug(frame,pSize=10,worm_sz=1000):
    ###high level function, takes in binarized frame and returns radial distances
    ###based on Ingmar and Jochen paper
    """
    #blur edges to smooth
    #median kernel
    kern=np.ones((3,3))
    kern[0,0]=0
    kern[0,2]=0
    kern[2,0]=0
    kern[2,2]=0
    manip=filters.median(frame,kern)
    #Gaussian and re-threshold
    manip=filters.gaussian(manip,3)
    manip[(manip<.1)]=0
    manip[(manip>0)]=1
    """
    manip=frame.copy()/np.max(frame)
    #if weird binarizing, skip
    if np.sum(manip)/manip.size>.15:
        print('binarize exception')
        return(np.ones(pSize)*np.nan)
    
    return radialShape_Debug(manip,pSize,worm_sz)

def comFromFrame(frame,pSize=10,worm_sz=1000):
    ###high level function, takes in binarized frame and returns com of worm object
    #blur edges to smooth
    #median kernel
    kern=np.ones((3,3))
    kern[0,0]=0
    kern[0,2]=0
    kern[2,0]=0
    kern[2,2]=0
    manip=filters.median(frame,kern)
    #Gaussian and re-threshold
    manip=filters.gaussian(manip,5)
    manip[(manip<.1)]=0
    manip[(manip>0)]=1
    #if weird binarizing, skip
    if np.sum(manip)/manip.size>.15:
        print('binarize exception')
        return(np.ones(pSize)*np.nan)
    #identify worm object
    lab=measure.label(frame)
    regions=measure.regionprops(lab)
    img=np.array([])
    for i,props in enumerate(regions):
        if props.convex_area<worm_sz:
            continue
        return props.centroid
    return (np.nan,np.nan)


##################
#Helper functions for making spline from skeleton.
#Skeletonization code in skelFuncs

def orderedCoord(S):
    blnk=np.zeros((1,2))
#    S=skeleton.copy()
    loc=findEndpoints(S)[0]
    coords=np.zeros((0,2))
    coords=np.append(coords,blnk,0)
    coords[-1,:]=loc
    S[loc]=0
    
    while np.sum(S)>0:
        loc=findNeighbor(S,loc)
        #check that there was a neighbor (solves disconnected graph bug)
        if not loc:
            break
        coords=np.append(coords,blnk,0)
        coords[-1,:]=loc
        S[loc]=0
    return coords

def findNeighbor(skeleton,loc):
    #finds next point on skeleton
#    print(loc)
    (x,y)=loc
    Z=[-1,0,1]
    for i in Z:
        for j in Z:
            if i==0 and j==0:
                continue
            if skeleton[x+i,y+j]>0:
                return (x+i,y+j)
    return False

def makeSpline(skeleton,nPoints=12):
    curve=orderedCoord(skeleton)
    if len(curve)<3: #check that we were able to make a continuous curve from skeleton
        return False
    kv=3
    if np.sum(skeleton<3):
        kv=2
        if np.sum(skeleton<2):
            kv=1
    tck,u=splprep([curve[:,0],curve[:,1]],s=10,k=kv)
    u=np.linspace(0,1,nPoints)
    return splev(u, tck)


##########################
#Functions that define parameters given a spline

def angleFromSpline(spline):
    #define angle of segments relative to horizontal
    dx=spline[0][1:]-spline[0][:-1]
    dy=spline[1][1:]-spline[1][:-1]
    ang=np.arctan(dy/dx)
    #return difference of adj angles
    return ang[1:]-ang[:-1]

def fftFromSpline(spline):
    #define complex signal
    C=spline[0]+(1j*spline[1])
    #FFT
    Q=np.fft.fft(C)
    #append real and complex part of fft and return
    #do appending here so can't mess up by operator
    return np.append(np.real(Q),np.imag(Q))
        
