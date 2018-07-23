import math
import numpy as np
import random as rand
from numpy import array as arr
from numpy import concatenate as cat

import scipy.io as sio
from scipy.misc import imread, imresize

class DataItem:
    pass

class DataHandler:
    def __init__(self,matfilepath,batchsize):
        self.datapath=matfilepath
        self.data = self._load_dataset()
        self.batchsize=batchsize
        self.numbatch=math.ceil(len(self.data)/self.batchsize)
        self.currbatch = 0

    def _load_dataset(self):
            # Load Matlab file dataset annotation
            mlab = sio.loadmat(self.datapath)
            raw_data = mlab
            # print(raw_data)
            mlab=mlab['data_compiler']
            num_images = mlab.shape[1]
            data = []

            sample = mlab[0,1]
            for i in range(num_images):
                sample = mlab[0, i]

                item = DataItem()
                item.image_id = i
                item.im_path = sample[0][0]
                item.zdepth = sample[1][0][0]
                data.append(item)

            rand.shuffle(data)
            return data

    def make_train_test_set(self,trainportion):
        cutidx=int(len(self.data)*trainportion)
        self.trainset = self.data[0:cutidx]
        self.testset = self.data[cutidx:]

    def get_nextbatch_fortraining(self):
        st_idx=self.batchsize*self.currbatch
        end_idx=self.batchsize*(self.currbatch+1)
        if end_idx>=len(self.trainset):
            end_idx=len(self.trainset)
            self.currbatch=0
        nextbatchdata = self.trainset[st_idx:end_idx]
        batch_img,batch_zdepth=self.retrieve_xydata(nextbatchdata)
        self.currbatch+=1
        return batch_img, batch_zdepth


    def retrieve_xydata(self,data):
        idx=0
        for item in data:
            #may have to replace '\' with '\\'
            img=imread(item.im_path)
            img=np.expand_dims(img,axis=0)
            img=np.expand_dims(img,axis=-1)

            zdepth=np.expand_dims(item.zdepth,axis=0)
            zdepth=np.expand_dims(zdepth,axis=1)
            if idx==0:
                batch_img=img
                batch_zdepth=zdepth
            else:
                batch_img=np.vstack((batch_img,img))
                batch_zdepth=np.vstack((batch_zdepth,zdepth))

            idx+=1
        return batch_img, batch_zdepth
        
    