import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import pickle
from scipy import ndimage
from dnn_app_utils import load_data, predict


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
num_px = train_x_orig.shape[1]

# Open trained model
parameters = pickle.load(open('pk','rb'))

## SELECT IMAGE ##
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END SELECT IMAGE ##

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))

my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
plt.show()
