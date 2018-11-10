#import sys
#sys.path.append("..")



#layer_output = calc_output(model, 'Dense1', 320, test)
def calc_output(model, layer_name, next_layer_index, train_dataset):
    from keras.models import  Model

    sample_num = len(train_dataset)
    
    weight = model.get_weights()
    w = weight[next_layer_index]   #w:(1024, 2)

    layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    layer_output_ = layer_model.predict(train_dataset)  #(1,1,1,1024)
    
    layer_output = []
    for i in range(0, sample_num):  #for each train sample:0-20000
        layer_result_one_sample = []
        layer_output_1 = layer_output_[i][0][0]
        
        for j in range(0, w.shape[1]):  #for each next layer node:0, 1
            layer_result_one_sample_one_node = []
            for k in range(0, w.shape[0]):  #for each this layer node: 0-1023
                #w:(1024, 2)
                layer_result_one_sample_one_node.append(layer_output_1[k]*w[k][j])
            layer_result_one_sample.append(layer_result_one_sample_one_node)
        layer_output.append(layer_result_one_sample)
    return layer_output  #(20000, 2, 1024)


def find_close_0_index(a_list, min_num):
    import heapq
    absMap = map(abs, a_list)
    absList = list(absMap)
    close_0_list = heapq.nsmallest(min_num, absList)
    min_index = set()
    for i in close_0_list:
        min_index.update([idx for idx,x in enumerate(absList) if x == i])
    #min_index = map(absList.index, heapq.nsmallest(min_num, absList))
    return list(min_index)

def evaluate_output(layer_output, most_close_0_num, most_times_close_0):
    from collections import Counter
    
    samples_num = len(layer_output)
    statistics_index_list = []
    
    for i in range(0, samples_num):   ##for each sample: 0- 20000
        for j in range(0 , len(layer_output[i])):  # for each node: 0,1
            close_0_index = find_close_0_index(layer_output[i][j], most_close_0_num)
            statistics_index_list+=close_0_index
    index_times_dic = Counter(statistics_index_list)
    dic = sorted(index_times_dic.items(), key = lambda index_times_dic:index_times_dic[1],reverse = True)
    most_close_0_idx = [i[0] for i in dic[0:most_times_close_0]]
    return most_close_0_idx


def pruning(model, layer_name, next_layer_index, train_dataset, most_close_0_num, most_times_close_0):
    from keras.models import  Model
    from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
    
    layer_output = calc_output(model, layer_name, next_layer_index, train_dataset)
    most_close_0_idx = evaluate_output(layer_output, most_close_0_num, most_times_close_0)
    model = delete_channels(model, model.get_layer(name=layer_name), most_close_0_idx)
    return model


def draw_weight(weight):
    import matplotlib.pyplot as plt

    plt.hist(weight)



#import math
#print((math.e ** x)/((math.e ** x)+(math.e ** y)))
#print((math.e ** y)/((math.e ** x)+(math.e ** y)))

#model = delete_channels(model, model.get_layer(name='dense2'), [0,4,67])







# delete layer_1 from a model
#model = delete_layer(model, layer_1)
# insert new_layer_1 before layer_2 in a model
#model = insert_layer(model, layer_2, new_layer_3)
# delete channels 0, 4 and 67 from layer_2 in model
#model = delete_channels(model, layer_2, [0,4,67])

#for layer in model.layers:
#    print(layer)
    #weights = layer.get_weights()
    #print(weights)
# delete layer_1 from a model
#model = delete_layer(model, model.get_layer(name='dense1'))
#model = delete_channels(model, model.get_layer(name='dense1'), [0,4,67])
#for layer in model.layers:
#    print(layer)

#json_string = model.to_json()
#open('alexNet_modify.json','w').write(json_string)
#model.save_weights('alexNet_modify.h5')



'''
weight = model.get_weights()
dense2_layer_model = Model(inputs=model.input,outputs=model.get_layer('dense2').output)
dense2_output = dense2_layer_model.predict(test)
w = weight[14]
bias = weight[15]
x = 0
for i in range(0,len(a)):
    x = x+ dense2_output[0][i]*w[i][0]
x = x+bias[0]
    
y= 0
for i in range(0,len(a)):
    y = y+ dense2_output[0][i]*w[i][1]
y = y+bias[1]
import math
print((math.e ** x)/((math.e ** x)+(math.e ** y)))
print((math.e ** y)/((math.e ** x)+(math.e ** y)))
'''

