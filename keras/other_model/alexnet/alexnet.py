import sys
sys.path.append("../..")
from util.dataset import get_files
    

def train_alexnet(train, train_label):
    import matplotlib.pyplot as plt
    from keras.callbacks import ReduceLROnPlateau, CSVLogger, TensorBoard
    from keras.models import Sequential  
    from keras.optimizers import SGD  
    from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
    from keras.layers import Convolution2D, MaxPooling2D
    from keras import backend as K
    
    model = Sequential()
    #???
    model.add(Convolution2D(filters=96, kernel_size=(11,11),
                     strides=(4,4), padding='valid',
                     input_shape=(227,227,3),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), 
                           strides=(2,2), 
                           padding='valid'))
    #???
    model.add(Convolution2D(filters=256, kernel_size=(5,5), 
                     strides=(1,1), padding='same', 
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), 
                           strides=(2,2), 
                           padding='valid'))
    #???
    model.add(Convolution2D(filters=384, kernel_size=(3,3), 
                     strides=(1,1), padding='same', 
                     activation='relu'))
    model.add(Convolution2D(filters=384, kernel_size=(3,3), 
                     strides=(1,1), padding='same', 
                     activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=(3,3), 
                     strides=(1,1), padding='same', 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), 
                           strides=(2,2), padding='valid'))
    #???
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
     
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
     
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
     
    # Output Layer
    model.add(Dense(2))
    model.add(Activation('softmax'))
     
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    #learnRate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    #log = CSVLogger('result/log_epoch.csv', separator=',', append=False)
    #board = TensorBoard(log_dir='result/logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    model.fit(train, train_label, batch_size=100, epochs=1, validation_split=0.1)#, callbacks=[log,board])
    model.save('alexnet.h5')
    


def test_alexnet():

    from keras.models import load_model
    import numpy as np
    import time

    test_dir= '/home/pdd/pdwork/CV_BiShe/picture/trainTest/'
    test, test_label = get_files(test_dir, 227, 227)
    TP = 0 
    TN = 0
    FP = 0
    FN = 0
    
    model = load_model('alexnet_2.h5')
    time_start=time.time()
    test_result = model.predict(test)
    time_end=time.time()
    time_spent = time_end-time_start
    print(time_spent)
    
    #calc precision and recall
    
    for i in range(0,len(test_label)):
        if np.argmax(test_result[i]) == 1:
            if np.argmax(test_label[i]) == 1:
                TP += 1
            else:
                FP += 1
        else:
            if np.argmax(test_label[i]) == 1:
                FN += 1
            else:
                TN += 1
    print(TP, TN, FP, FN)
    precision = TP/(FP+TP)
    recall    = TP/(FN+TP)
    print('precision', precision)
    print('recall', recall)
    return time_spent, precision, recall


#train_dir ="/home/pdd/pdwork/CV_BiShe/picture/picTrain"
train_dir ="/home/pdd/pdwork/CV_BiShe/picture/trainTest"
train, train_label = get_files(train_dir, 227, 227)
train_alexnet(train, train_label)
