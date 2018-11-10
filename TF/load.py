from PIL import Image
import tensorflow as tf
import mobilenet_v2
import numpy as np
import os
import os.path
import time

def get_one_image(pic):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    image = Image.open(pic)
    image = image.resize([224, 224])
    image = np.array(image)
    return image

def evaluate_one_image(pic):
    '''Test one image against the saved models and parameters
    '''
    image_array = get_one_image(pic)

#    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 224, 224, 3])
        logit = mobilenet_v2.mobilenetv2(image, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[224, 224,3])
        
        # you need to change the directories to yours.
        logs_train_dir = '/home/pdd/pdwork/CV_BiShe/compression/mobileNetV2_TF/myself/log/' 
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
#                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            #if max_index==0:
             #  print('This is a clear with possibility %.6f' %prediction[:, 0])
            #else:
            #   print('This is a msk with possibility %.6f' %prediction[:, 1])
            return max_index

rootdir= '/home/pdd/pdwork/CV_BiShe/picture/trainTest/'

pic = os.listdir(rootdir)
print("checking the folder %s, there are %d pictures"%(rootdir,len(pic)))
print("---------------------------------------------------------")
time_start=time.time()
for i in range(0,len(pic)):
        imagepath = os.path.join(rootdir,pic[i])
        print("checking the pic %s,this is the %d picture"%(imagepath,i))
        result=evaluate_one_image(imagepath)
        print(result)

time_end=time.time()
time_spent = time_end-time_start
print(time_spent)
