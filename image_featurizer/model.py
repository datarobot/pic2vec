from keras.applications.xception import Xception







model = Xception(include_top = False, weights = None)
model.load_weights('../models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
