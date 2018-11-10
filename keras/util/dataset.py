
def get_files(file_dir, img_rows, img_cols):
    import os
    from PIL import Image
    import numpy as np
    import tflearn
    
    train = []
    label= []
    for filename in os.listdir(file_dir):
        name = filename.split('.')
        image = Image.open(os.path.join(file_dir, filename))
        image = image.resize([img_rows, img_cols])
        image = np.array(image)
        train.append(image/255)
        if name[0]=='0':
            label.append(0)
        else:
            label.append(1)
    print('There are %d pic'%(len(label)))
    train = np.array(train)
    label = tflearn.data_utils.to_categorical(label, 2)
    return train,label
