#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:15:17 2021

@author: sam
"""

#Parse inputs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("file", help="path to video file")
parser.add_argument("-intenseSample",type=float,
                    default=40, help="minutes to test intensity for")
parser.add_argument("-manualLight",action='store_true',help="manually define light ROI")
parser.add_argument("-invert",action='store_true',
                    default=False, help="invert image when using brightfield imaging")
parser.add_argument("-ignoreFR",action='store_true',
                    default=False, help="dont test framerate==2")
parser.add_argument("-initFrame",type=int,
                    default=20, help="initial frame to identify worms on")
parser.add_argument("-background",type=str,
                    default='', help="path of image files to average and use as background removal")
parser.add_argument("-zeroCenter",action='store_true',
                    default=False, help="temp081822")
args = parser.parse_args()
vidName=args.file

#load needed packages
import os, sys
from wormParam import *
from wormparam_roiFree import extractAllWorms_ROIfree
from imgCalibration import *
import numpy as np
import skvideo.io
from skimage.filters import threshold_otsu
import time
import pickle

# check that file is good
VIDEOS = [vidName]
FOLDERS = [f'pickles{vidName[vidName.rfind("/"):vidName.rfind(".mp4")]}/']
print(FOLDERS)
PASS = True
for i,vidName_i in enumerate(VIDEOS[:]):
    #deal with any frame rate issues from recording
    if not skvideo.io.ffprobe(vidName_i)['video']['@r_frame_rate']=='2/1' and not args.ignoreFR:
        print('FRAME RATE ERROR: ', vidName_i)
        # exit()
        nm = vidName[:vidName.rfind(".mp4")]
        resolved=False
        #try trimming the first few minutes of video
        for j in range(30):
            command = f'ffmpeg -ss 00:{int(j):02}:00.0 -i {nm}.mp4 -c copy -t 00:01:10.0 -metadata r_frame_rate=2/1 {nm}_fixed.mp4'
            os.system(command)
            print('FRAME RATE: ',skvideo.io.ffprobe(nm+'_fixed.mp4')['video']['@r_frame_rate'])
            #if successful, re-render without those timepoints
            if skvideo.io.ffprobe(nm+'_fixed.mp4')['video']['@r_frame_rate']=='2/1':
                resolved = True
                print(f'Resolved frame rate error at {j} min')
                command = f'ffmpeg -ss 00:{int(j):02}:00.0 -i {nm}.mp4 -c copy -metadata r_frame_rate=2/1 {nm}_fixed.mp4'
                os.system(command)
                break
        if not resolved:
            print('ERROR: Failed to resolve frame rate error in {nm}')
            exit()
        else:
            VIDEOS[i] = nm+'_fixed.mp4'
            
# calculate background frame for removal
if len(args.background)>0:
    from skimage.filters import gaussian
    from skimage import io
    print('calculating background')
    if os.path.isfile(os.listdir(args.background)[1]):
        tiffStack = sorted(os.listdir(args.background))
    else:
        tiffStack = []
        folder = sorted(os.listdir(args.background))
        for f in folder:
            if f == 'location/':
                continue
            if not os.path.isdir(args.background+f+'/'):
                continue
            tiffStack.extend([f+'/'+file for file in sorted(os.listdir(args.background+f))])
    background_list=[]
    for i,file in enumerate(tiffStack):
        if i%30>0: continue
        background_list.append(io.imread(args.background+file))
        if len(background_list)>50:
            break
    background_list=np.array(background_list)
    # background = np.median(background_list,axis=0)
    background = np.mean(background_list,axis=0)
    plt.imshow(background)
    clicked_m = plt.ginput(n=-1, timeout=-1)
    plt.close('all')
else:
    background = False

            
vidName=VIDEOS[0]            
#get worm locations
temp = skvideo.io.vread(vidName,num_frames=args.initFrame)[:,:,:,0]
if len(args.background)>0:
    temp = temp.astype(float) -background[None,:,:]
# temp = temp[:,:,:,None]   
#080922
# if args.invert:
#     temp=-temp+255
plt.imshow(temp[-1,...]) #shows full framedark
plt.title('Click the worms')
clicked_m = plt.ginput(n=-1, timeout=-1)
plt.close('all')
COM_init = [(point[1],point[0]) for point in clicked_m]

# get light location
if args.manualLight:
    plt.imshow(temp[-1,...])
    plt.show(block=False)
    plt.title('Define indicator position (manual)')
    top = float(input('Top of indicator: '))
    bottom = float(input('Bottom of indicator: '))
    left = float(input('Left of indicator: '))
    right = float(input('Right of indicator: '))
    plt.close('all')
    ROI_intense=((int(max(top,0)),int(min(bottom,temp.shape[1]))),
                 (int(max(left,0)),int(min(right,temp.shape[2])))) 
else:
    l_sz=100
    plt.imshow(temp[-1,...]) #shows full framedark
    plt.title('Click the indicator light')
    light_loc = plt.ginput(n=1, timeout=-1)[0]
    plt.close('all')
    ROI_intense=((int(max(light_loc[1]-l_sz,0)),int(min(light_loc[1]+l_sz,temp.shape[1]))),
                 (int(max(light_loc[0]-l_sz,0)),int(min(light_loc[0]+l_sz,temp.shape[2])))) 

#get intensity
n=120*args.intenseSample
intense=intensityTimeseries(vidName, n, ROI_intense)
plt.plot(intense)
plt.title(vidName)
plt.show(block=False)
LD_thresh = float(input('Light Dark Threshold:'))
plt.close('all')

# Check threshold and ROI PARAMS
roi_size=130
size_lim=120
thresh_scale=.83
frame=temp[-1,:,:]
if len(frame.shape)==2:
    frame=frame[:,:,None]
ROI_all=[]
for com in COM_init:
    ROI_all.append(((int(max(com[0]-roi_size,0)),int(min(com[0]+roi_size,frame.shape[0]))),
                   (int(max(com[1]-roi_size,0)),int(min(com[1]+roi_size,frame.shape[1])))))

redo = True
while redo:
    fig, ax = plt.subplots(nrows=2,ncols=len(ROI_all),sharex='col',sharey='col', figsize=(16,8))
    print('worm sizes')
    for i in range(len(ROI_all)):
        ROI=ROI_all[i]
        ang_len=100
        if args.invert:
            bi_thresh=thresh_scale*np.percentile(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0],.1)
            img=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0]<bi_thresh
        else:
            bi_thresh=thresh_scale*np.percentile(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0],99.9)
            img=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0]>bi_thresh
        # if args.invert:
        #     raw_image=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0].copy()
        #     if args.zeroCenter: raw_image-raw_image.mean()
        #     bi_thresh=thresh_scale*np.percentile(raw_image,.1)
        #     img=raw_image<bi_thresh
        # else:
        #     bi_thresh=thresh_scale*np.percentile(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0],99.9)
        #     img=frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0]>bi_thresh
        temp2=radialShapeFromFrame(img,ang_len,worm_sz=size_lim,com=True)
        com=temp2[1]
        
        im=ax[0,i].imshow(frame[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],0])
        img=filters.gaussian(img,3)
        img=img>.5
        print(img.sum())
        ax[1,i].imshow(img)
        ax[1,i].scatter(com[1],com[0],c='r',s=5)
    plt.show(block=False)
    redo = False
    val = input(f'worm size threshold (current={size_lim}):')
    if not (val is ''):
        size_lim = float(val)
        redo=True
    val = input(f'binary threshold scale (current={thresh_scale}):')
    print(val)
    if not (val is ''):
        thresh_scale = float(val)
        redo=True
    plt.close('all')
    
# Run segmentation
for vidName, folder in zip(VIDEOS[:], FOLDERS[:]):
    print(vidName)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    binarize_percentile=99.9
    if args.invert:
        binarize_percentile=.1
    COM_init = extractAllWorms_ROIfree(vidName,folder,COM_init,save=30000,roi_size=roi_size,com=True,
                intense_region=ROI_intense,LD_thresh=LD_thresh,
                size_lim=[size_lim,size_lim],thresh_scale=thresh_scale,
                invert=args.invert,binarize_percentile=binarize_percentile,
                background=background,)    


# Make and save log files
log_ = {'videos':VIDEOS,
        'folders':FOLDERS,
        'LD_thresh': LD_thresh,
        'worms': COM_init,
        'indicator': ROI_intense,
        'roi_size': roi_size,
        'size_lim':size_lim,
        'thresh_scale':thresh_scale,
        }
with open(f'segmenting/{vidName[vidName.rfind("/"):vidName.rfind(".mp4")]}.pickle','wb') as f:
    pickle.dump(log_,f)

        


