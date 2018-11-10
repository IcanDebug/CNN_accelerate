#coding=utf-8  
import sys
sys.path.append("../..")
from util.dataset import get_files
from keras.models import Model  
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D  
from keras.layers import add,Flatten  
  
def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  
  
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')  
    if with_conv_shortcut:  
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)  
        x = add([x,shortcut])  
        return x  
    else:  
        x = add([x,inpt])  
        return x  
def train_resnet34(train, train_label):
    from keras.callbacks import ReduceLROnPlateau, CSVLogger, TensorBoard
    import matplotlib.pyplot as plt
    
    inpt = Input(shape=(224,224,3))  
    x = ZeroPadding2D((3,3))(inpt)  
    x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')  
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  
    #(56,56,64)  
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))  
    #(28,28,128)  
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))  
    #(14,14,256)  
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))  
    #(7,7,512)  
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)  
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))  
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))  
    x = AveragePooling2D(pool_size=(7,7))(x)  
    x = Flatten()(x)  
    x = Dense(2,activation='softmax')(x)  
      
    model = Model(inputs=inpt,outputs=x)  
    model.compile(loss="categorical_crossentropy",optimizer='sgd',metrics=['accuracy'])
    #learnRate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    log = CSVLogger('result/log_epoch.csv', separator=',', append=False)
    board = TensorBoard(log_dir='result/logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    history=model.fit(train, train_label, batch_size=100, epochs=20, validation_split=0.1, callbacks=[log,board])
    
    plt.plot()  
    # summarize history for loss  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()  
    
    model.save('resnet34.h5')

def test_resnet34():

    from keras.models import load_model
    import numpy as np
    import time

    time_start=time.time()
    test_dir= '/home/pdd/pdwork/CV_BiShe/picture/trainTest/'
    test, test_label = get_files(test_dir, 224, 224)
    TP = 0 
    TN = 0
    FP = 0
    FN = 0
    
    model = load_model('resnet34.h5')
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
train, train_label = get_files(train_dir, 224, 224)
train_resnet34(train, train_label)