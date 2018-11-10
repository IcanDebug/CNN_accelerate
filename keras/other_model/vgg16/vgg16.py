#coding=utf-8  
import sys
sys.path.append("../..")
from util.dataset import get_files

def train_vgg16(train, train_label):
    from keras.models import Sequential  
    from keras.layers import Dense,Flatten,Dropout  
    from keras.layers.convolutional import Conv2D,MaxPooling2D  
    import numpy as np
    from keras import backend as K
    from keras.callbacks import ReduceLROnPlateau, CSVLogger, TensorBoard
    import matplotlib.pyplot as plt
      
    model = Sequential()
    if K.image_dim_ordering() == 'th':
        input_shape = (3,224,224)
    else:
        input_shape = (224,224,3)
        
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=input_shape,padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Flatten())  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(1000,activation='softmax'))  
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
    model.summary()
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
    model.save('vgg16.h5')

def test_vgg16(train, train_label):
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
    
    model = load_model('vgg16.h5')
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