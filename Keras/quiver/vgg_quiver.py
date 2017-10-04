from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet', include_top=False)
from quiver_engine.server import launch
launch(model, input_folder='./imgs', port=7000)

