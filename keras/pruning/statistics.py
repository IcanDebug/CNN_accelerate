#from keras.layers import Convolution2D
#from keras.models import load_model
#from keras.utils.generic_utils import CustomObjectScope
#import keras
#model = load_model(h5)
#with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
#        model = load_model('demo.h5')


#w = statistics_weight(model)
def statistics_weight(model):
    weights = {}
    for layer in model.layers:
        layer_weight = layer.get_weights()
        if isinstance(layer, Convolution2D):
            weights[layer.name+'/kernel'] = layer_weight[0]
            weights[layer.name + '/bias'] = layer_weight[1]
    return weights

#weight_cnn = show_cnn_weight(w)
def show_cnn_weight(weights):
    weight_cnn = []
    for key in weights:
        if 'conv2d' in key and 'depthwise' not in key and 'kernel' in key:
            for i in weights[key]:
                for j in i:
                    for k in j:
                        for l in k:
                            weight_cnn.append(l)
    return weight_cnn
  
#parzen
def draw_pro_distribution(weight_cnn):
    import matplotlib.pyplot as plt
    plt.xlabel("weight")
    plt.ylabel("numbers of parameter")
    plt.hist(weight_list,50)
    
    
    
def draw(weight_list):
    from scipy import stats
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    
    # plot histogram
    plt.subplot(221)
    plt.hist(weight_list, 50)
    
    # obtain histogram data
    plt.subplot(222)
    hist, bin_edges = np.histogram(weight_list)
    plt.plot(hist)
    
    # fit histogram curve
    plt.subplot(223)
    sns.distplot(weight_list, kde=False, fit=stats.gamma, rug=True)
    plt.show()
    
def draw_cdf(weight_cnn):
    from scipy import stats
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    weight_list = [abs(i) for i in  weight_cnn]
    
    
    plt.subplot(121)
    hist, bin_edges = np.histogram(weight_list)
    cdf = np.cumsum(hist)
    plt.plot(cdf)
    
    plt.subplot(122)
    cdf = stats.cumfreq(weight_list)
    plt.plot(cdf[0])
    
    plt.show()
    
def draw_cdf_pdf(weights):
    from scipy import stats
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    
    hist, bin_edges = np.histogram(weights)
    width = (bin_edges[1] - bin_edges[0]) * 0.8
    plt.bar(bin_edges[1:], hist/max(hist), width=width, color='#5B9BD5')
    
    cdf = np.cumsum(hist/sum(hist))
    plt.plot(bin_edges[1:], cdf, '-*', color='#ED7D31')
    
    plt.xlim([-2, 2])
    plt.ylim([0, 1])
    plt.grid()
    
    plt.show()

#x, p = parzen2(weight_cnn,0.001)
def parzen2(input_,h):
    import  math
    
    x=list(range(-1000,1000))
    x=[i/1000 for i in x]
    
    p = [0] * len(x)
    count = 0
    for i in input_:
        count += 1
        print(count)
        for j in range(len(x)):
            p[j] += math.exp(((x[j]-i)/h)**2/(-2))/math.sqrt(2*math.pi)/h
    p_list = [i/len(input_) for i in p]
    return x, p_list 
            

def plot_weight(x, p):
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.xlabel("weight")
    plt.ylabel("numbers of parameter")
    
    #sort_index = np.argsort(x)
    #x1=[x[i] for i in sort_index]
    #p1=[p[i] for i in sort_index]
    plt.plot(x, p)
    plt.show()


def save_var(var, pkl_name):
    import pickle
    with open(pkl_name, 'wb') as f:
        pickle.dump(var, f)
        
def load_var(pkl_name):
    import pickle
    with open(pkl_name, 'rb') as f:
        aa = pickle.load(f)
    return aa