import numpy as np
import time
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
import keras
from util.dataset import get_files

test_dir= '/home/pdd/pdwork/CV_BiShe/picture/trainTest/'
test, test_label = get_files(test_dir, 224, 224)

def test_model(test, test_label, model_name):
    #test_dir= '/home/pdd/pdwork/CV_BiShe/picture/picValidation/'

    TP = 0 
    TN = 0
    FP = 0
    FN = 0
    
    if(model_name[-3:] != '.h5'):
        model_name += '.h5'
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(model_name)#('DenseDWNet.h5')
    time_start=time.time()    
    test_result = model.predict(test)
    time_end=time.time()
    time_spent = time_end-time_start
    print('model name:', model_name[:-3])
    print('time spent(s) :', time_spent)
    
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
#test_model(test, test_label)


#for i in range(0,5):
##    test_model(test, test_label, 'demo')
#for i in range(0,5):
#    test_model(test, test_label, 'round')





'''
#print("checking the folder %s, there are %d pictures"%(rootdir,len(pic)))
print("%d pictures"%len(pic))
print("---------------------------------------------------------")
for i in range(0,len(pic)):
    imagepath = os.path.join(rootdir,pic[i])
    print(pic[i])
        #print("checking the pic %s,this is the %d picture"%(imagepath,i))
    result=predicMsk(imagepath,model)
    print(result)
    if result == 1:
        if int(pic[i].split('.')[0]) == 1:
            TP += 1
        else:
            FP += 1
    else:
        if int(pic[i].split('.')[0]) == 1:
            FN += 1
        else:
            TN += 1
        #savepath=os.path.join(savedir,pic[i])
        #img = Image.open(imagepath)
        #img.save(savepath)

    print("---------------------------------------------------------")
time_end=time.time()
print(time_end-time_start)
print('precision', TP/(FP+TP))
print('recall', TP/(FN+TP))
'''
