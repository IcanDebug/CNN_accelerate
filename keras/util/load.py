#model = load_model()
def load_model(h5_path='../mobileNetv2.h5'):
    from keras.utils.generic_utils import CustomObjectScope
    import keras

    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(h5_path)
    return model