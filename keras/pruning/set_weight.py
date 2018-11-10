#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 20:23:26 2018

@author: pdd
"""
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
import keras

#w1 = set_weight(model)
def set_weight(model):
    import numpy as np
    w = model.get_weights()
    for i in range(len(w)):
        if len((w[i]).shape)==1:
            for j in range(len(w[i])):
                w[i][j] = np.float16(w[i][j])
        elif len((w[i]).shape)==4:
            for a in range(len(w[i])):
                for b in range(len(w[i][a])):
                    for c in range(len(w[i][a][b])):
                        for d in range(len(w[i][a][b][c])):
                            w[i][a][b][c][d] = np.float16(w[i][a][b][c][d])
        else:
            for a in range(len(w[i])):
                for b in range(len(w[i][a])):
                    w[i][a][b] = np.float16(w[i][a][b])
    return w


#with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
#    model = load_model('demo.h5')

#w1 = set_weight(model)
#model.set_weights(w1)
#w =  model.get_weights()
#model.save('round.h5')