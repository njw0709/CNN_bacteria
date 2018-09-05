import tensorflow as tf
from classify import Classify
import os
import scipy.io as sio
from skimage import io
import matplotlib.pyplot as plt
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class DataItem:
    pass

def load_dataset(matfilepath):
    # Load Matlab file dataset annotation
    mlab = sio.loadmat(matfilepath)
    mlab = mlab['data_compiler']
    num_images = mlab.shape[1]
    testdata = []

    for i in range(num_images):
        sample = mlab[0, i]

        item = DataItem()
        item.image_id = i
        item.im_path = sample[0][0]

        item.zdepth = sample[1][0][0]
        testdata.append(item)

    return testdata

def drawplots(image,zpred,answer,nums,signal,axes):

    axes[0].imshow(image,cmap='gray',vmin=66, vmax=85)
    axes[0].set_title('Measurement')
    axes[1].imshow(zpred,cmap='gray',vmin=17000, vmax=22000)
    axes[1].set_title('Prediction Z='+str(nums[0]))
    axes[2].imshow(answer,cmap='gray',vmin=17000, vmax=22000)
    axes[2].set_title('Answer Z=' + str(nums[1]))
    z = list(range(1, 2001))
    axes[3].plot(z, signal)

testing = True
matfilepath = '..\\img_class_png.mat'
resnettest = Classify()
resnettest.initialize()
data = load_dataset(matfilepath)
refstackpath ='D:\\Katja\\ReferenceStacks\\bin2x2\\H2V2Z1_FinalRefstack_allbeads_latdrift_corr_min20.tif'
refimgstack=io.imread(refstackpath)
# testing code
for datum in data:
    image = io.imread(datum.im_path)
    zpospred, signal =resnettest.predictZ(image)
    refimg=refimgstack[zpospred[0],:,:]
    answer=refimgstack[datum.zdepth,:,:]
    fig,(ax1,ax2,ax3)=plt.subplots(1,3)
    fig2,ax4=plt.subplots()
    axes=[ax1,ax2,ax3,ax4]
    nums=[zpospred[0],datum.zdepth]
    drawplots(image,refimg,answer,nums,signal,axes)


