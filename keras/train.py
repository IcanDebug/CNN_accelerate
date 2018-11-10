from model.mobilenet_v2 import MobileNetv2
from model.proposed import DenseDWNet
import matplotlib.pyplot as plt
from util.dataset import get_files
from keras.callbacks import ReduceLROnPlateau, CSVLogger, TensorBoard,ModelCheckpoint,LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


train_dir ="/home/pdd/pdwork/CV_BiShe/picture/picTrain"
#train_dir ="/home/pdd/pdwork/CV_BiShe/picture/picTrain"
x_train, y_train = get_files(train_dir, 224, 224)

validation_dir = "/home/pdd/pdwork/CV_BiShe/picture/picValidation"
x_test, y_test = get_files(train_dir, 224, 224)
#model = MobileNetv2((224, 224, 3), 2)

model = DenseDWNet((224, 224, 3), 2)

model.compile(loss="categorical_crossentropy",optimizer='sgd',metrics=['accuracy'])
log = CSVLogger('result/log_epoch.csv', separator=',', append=False)
board = TensorBoard(log_dir='result/logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
filepath="model_{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                        mode='max')

def step_decay(epoch):
    import math
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)

train_datagen = ImageDataGenerator(
    #rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
#test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(x_train, y_train,
        batch_size=32) 
#validation_generator = test_datagen.flow(x_test, y_test,
#    batch_size=32)

history=model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    nb_epoch=50,
    validation_data=(x_test, y_test),
    callbacks=[log,board, lrate,checkpoint])

plt.plot()  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
#plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()  

model.save('DenseDWNet_Aug.h5')

'''
from keras.utils.generic_utils import CustomObjectScope
import keras
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model1 = load_model('network.h5')
'''
'''
test = []
image = Image.open('/home/pdd/pdwork/CV_BiShe/picture/trainTest/0.0.jpg')
image = image.resize([224, 224])
image = np.array(image)
test.append(image/255)
test = np.array(test)

weight = model.get_weights()
dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense1').output)
dense1_output = dense1_layer_model.predict(test)
'''