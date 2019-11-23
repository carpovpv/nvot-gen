

import sys
import os
import uuid
import time
from functools import partial
import math
import glob
import _pickle as cPickle
import h5py
import random
import numpy as np
from numpy import genfromtxt
from collections import namedtuple;
from tqdm import tqdm

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3";

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
#from tensorflow.keras.utils import plot_model

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import socket
import sys
import struct

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False);
config.gpu_options.allow_growth = True;
tf.logging.set_verbosity(tf.logging.ERROR);
K.set_session(tf.Session(config=config));

source_t = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
source_t.connect(("127.0.0.1", 6222));

try:
   source_v = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
   source_v.connect(("127.0.0.1", 6223));
except:
   pass;
 
def data_generator(mode):

   batches = 10;
   while True:

      if mode == "t": source_t.send(mode.encode());
      else : source_v.send(mode.encode());

      data = bytearray();
      while len(data) < 17039400:
         if mode == "t":
            p = source_t.recv(65536);        
         else:
            p = source_v.recv(65536);
         data.extend(p);

      frmt = "=" + str(batches * (32 * 32 * 32 * 13 + 1) ) + "f";
      f = struct.unpack(frmt, data);
      arr = np.array(f);
      y = arr[-batches:].reshape(10,1);
      x = arr[:-batches].reshape((batches,32,32,32,13));

      yield [x], y;

def buildModel():

   l_in = layers.Input((32,32,32,13,), name = "Input");

   #first inseption module
   l_ins_0 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(1,1,1) )(l_in);

   l_ins_1 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(1,1,1) )(l_in);
   l_ins_1 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(3,3,3) )(l_ins_1);

   l_ins_2 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(1,1,1) )(l_in);
   l_ins_2 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(5,5,5) )(l_ins_2);

   l_ins_3 = layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(1,1,1), padding ='same')(l_in);
   l_ins_3 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(1,1,1) )(l_ins_3);

   l_ins = layers.Concatenate()([l_ins_0, l_ins_1, l_ins_2, l_ins_3]);
   l_ins = layers.Dropout(rate=0.1)(l_ins);

   l_in_2 = layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(1,1,1))(l_ins);
   l_in_2 = layers.BatchNormalization()(l_in_2);

   #second inseption module
   l_ins_0 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(1,1,1) )(l_in_2);

   l_ins_1 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(1,1,1) )(l_in_2);
   l_ins_1 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(3,3,3) )(l_ins_1);

   l_ins_2 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(1,1,1) )(l_in_2);
   l_ins_2 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(5,5,5) )(l_ins_2);

   l_ins_3 = layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(1,1,1), padding ='same')(l_in_2);
   l_ins_3 = layers.Conv3D(filters = 32, padding= 'same', activation='relu', kernel_size =(1,1,1) )(l_ins_3);

   l_ins2 = layers.Concatenate()([l_ins_0, l_ins_1, l_ins_2, l_ins_3, l_in_2]);
   l_ins2 = layers.Dropout(rate=0.1)(l_ins2);

   l_enc = layers.GlobalMaxPooling3D()(l_ins2);

   #dense layers
   l_enc = layers.Dense(128, activation='relu')(l_enc);
   l_enc = layers.BatchNormalization()(l_enc);
   l_enc = layers.Dropout(rate=0.1)(l_enc);

   l_enc = layers.Dense(64, activation='relu')(l_enc);
   l_enc = layers.Dropout(rate=0.1)(l_enc);

   #to binary loss
   l_out = layers.Dense(1, activation='sigmoid') (l_enc);

   mdl = tf.keras.Model([l_in], l_out);
   mdl.compile (optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc']);

   mdl.summary();
   #plot_model(mdl, to_file='nvot.png', show_shapes= True);

   return mdl;


mdl = buildModel();

'''
mdl.load_weights("tr-4.h5");

fp = open("res","w");
for i in range(2994):
   x,y = next(data_generator("t"));
   x_ = np.mean(mdl.predict(x));
   y_ = y.mean();
   print(y_, x_, file=fp);
   print(y_, x_);

fp.close();
sys.exit(0);
'''

#these values depend on the target!
NTRAIN = 25356;
NVALID = 4475;

class GenCallback(tf.keras.callbacks.Callback):
   def on_epoch_end(self, epoch, logs={}):
       mdl.save_weights("tr-" + str(epoch) + ".h5", save_format="h5");

history = mdl.fit_generator( generator = data_generator("t"),
                             steps_per_epoch = NTRAIN,
                             epochs = 5,
                             use_multiprocessing=False,
                             shuffle = False,
                             validation_data = data_generator("v"),
                             validation_steps = NVALID, callbacks = [ GenCallback() ]);



