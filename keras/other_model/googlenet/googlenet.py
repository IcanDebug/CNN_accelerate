import sys
sys.path.append("../..")
from util.dataset import get_files
from keras.models import Sequential, Model
from keras.optimizers import SGD  
from keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, normalization


def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Convolution2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = normalization.BatchNormalization(axis=3,name=bn_name)(x)
    return x

def Inception(x,nb_filter):
    branch1x1 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)

    branch3x3 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    branch3x3 = Conv2d_BN(branch3x3,nb_filter,(3,3), padding='same',strides=(1,1),name=None)

    branch5x5 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    branch5x5 = Conv2d_BN(branch5x5,nb_filter,(1,1), padding='same',strides=(1,1),name=None)

    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branchpool = Conv2d_BN(branchpool,nb_filter,(1,1),padding='same',strides=(1,1),name=None)

    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)

    return x

def train_googlenet(train, train_label):
    import matplotlib.pyplot as plt
    from keras.callbacks import ReduceLROnPlateau, CSVLogger, TensorBoard
    
    
    inpt = Input(shape=(224, 224, 3)) 
    x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Conv2d_BN(x,192,(3,3),strides=(1,1),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,64)#256
    x = Inception(x,120)#480
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,128)#512
    x = Inception(x,128)
    x = Inception(x,128)
    x = Inception(x,132)#528
    x = Inception(x,208)#832
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,208)
    x = Inception(x,256)#1024          
    x = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x)
    
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000,activation='relu')(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inpt,x,name='inception')
    
    model.compile(loss="categorical_crossentropy",optimizer='sgd',metrics=['accuracy'])
    #learnRate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    log = CSVLogger('result/log_epoch.csv', separator=',', append=False)
    board = TensorBoard(log_dir='result/logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    history=model.fit(train, train_label, batch_size=100, epochs=50, validation_split=0.1, callbacks=[log,board])
    
    plt.plot()  
    # summarize history for loss  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()  

    model.save('googlenet.h5')
    
   
#p = test_googlenet()
def test_googlenet():
    from keras.models import load_model
    import numpy as np
    import time

    test_dir= '/home/pdd/pdwork/CV_BiShe/picture/trainTest/'
    test, test_label = get_files(test_dir, 224, 224)
    TP = 0 
    TN = 0
    FP = 0
    FN = 0
    
    model = load_model('googlenet.h5')
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
    precision = TP/(FP+TP)
    recall    = TP/(FN+TP)
    print('precision', precision)
    print('recall', recall)
    return time_spent, precision, recall

train_dir ="/home/pdd/pdwork/CV_BiShe/picture/picTrain"
#train, train_label = get_files(train_dir, 224, 224)
test_googlenet()
