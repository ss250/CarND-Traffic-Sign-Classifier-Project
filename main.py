# Load pickled data
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, SGD
from keras.layers import *
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import time
from keras.utils import to_categorical
from tensorflow.python.client import device_lib

from keras import backend as K

# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
validation_file = "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(y_train)

# TODO: Number of validation examples
n_validation = len(y_valid)

# TODO: Number of testing examples.
n_test = len(y_test)

# TODO: What's the shape of an traffic sign image?
image_shape = np.array(X_train[0]).shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_test))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image)
print(y_train[index])


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.


X_train, y_train = shuffle(X_train, y_train)


### Define your architecture here.
### Feel free to use as many code cells as needed.


class ModelLoader():

    def __init__(self, n_labels, model_name,
                 saved_weights=None, optimizer=None, image_size=(100, 176)):

        self.n_labels = n_labels
        self.load_model = load_model
        self.saved_weights = saved_weights
        self.model_name = model_name
        image_size = image_size[::-1] # bit of a hack

        # Loads the specified model
        if self.model_name == 'small_c2d':
            print('Loading Small C3D model')
            self.input_shape = image_size + (3,)
            self.model = self.small_c2d()


        metrics = ['accuracy', 'top_k_categorical_accuracy']

        # If no optimizer is given, use Adam as default
        if optimizer is None:
            optimizer = Adam()

        self.model.compile(loss='categorical_crossentropy',
        # self.model.compile(loss='mean_squared_error',
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def small_c2d(self):
        # Tunable parameters

        model = Sequential()

        # Conv Block 1
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv Block 2
        model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv Block 2
        model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Dense Block
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.n_labels, activation='softmax'))

        return model
optimizer = SGD(lr=0.001)
# optimizer = Adam()
ml = ModelLoader(n_classes, 'small_c2d', image_size=image_shape[0:2], optimizer = optimizer)
model = ml.model

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
#Define callbacks


batch_size = 10
n_epochs = 10
steps_per_epoch = n_train // batch_size


model_name = 'c2d'
checkpointer = ModelCheckpoint(
    filepath=model_name + '-{epoch:03d}-{loss:.3f}.hdf5',
    verbose=1,
    save_best_only=True)
early_stopper = EarlyStopping(patience=2)

callbacks = [early_stopper,checkpointer]

# Training
print('Starting training')

y_train_cat = to_categorical(y_train, n_classes)
y_valid_cat = to_categorical(y_valid, n_classes)

history = model.fit(
    x=X_train,
    y=y_train_cat,
    epochs=n_epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=(X_valid, y_valid_cat),
    batch_size=batch_size
)


# model.save("./" + model_name + "_" + +  ".h5")
model.save("./{}_{}.h5".format(model_name, str(n_train)))
