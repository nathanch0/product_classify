import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math



# Image sizes
img_width = 224
img_height = 224

# These model weights are downloaded from github repo for keras
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_path = 'Training_Data/'
validation_data_path = 'Validation_Data/'

# Epochs to train top model
epochs = 50

# Batch sizes
batch_size = 16

def make_save_bottleneck_features():
    """
    This function will build out our bottleneck features from our training data
    using the weights that were established by VGG-16 (Transfer learning)
    """
    # build VGG16 network. We do not want to include the final fully-connected layers
    # and we load the Imagenet weights
    model = applications.VGG16(include_top=False, weights='imagenet')

    data_generator = ImageDataGenerator(rescale=1./255)

    # Make bottleneck features for training set
    generator = data_generator.flow_from_directory(train_data_path,
                                                   target_size = (img_width, img_height),
                                                   batch_size = batch_size,
                                                   class_mode = None,
                                                   shuffle = False)

    # print(len(generator.filenames)) #This shows the number of files in train_data
    # print(generator.class_indices) #This shows a dictionary of the titles of the sub-directories and its coorisponding number
    # print(len(generator.class_indices)) #This shows the number of classes in total (63)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(generator, predict_size_train)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    # Make bottleneck features for validation set
    generator_val = data_generator.flow_from_directory(validation_data_path,
                                                       target_size=(img_width,img_height),
                                                       batch_size=batch_size,
                                                       class_mode=None,
                                                       shuffle=False)

    nb_validation_samples = len(gnerator_val.filenames)

    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_venerator(generator_val, predict_size_validation)

    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)


def train_top_model():
    """
    This function will load the saved bottleneck features that were established from applying transfer learning
    onto the train and validation data. The function will use these features to train a fully connected network
    with its output as our desired classes
    """
    train_data = np.load('bottleneck_features_train.npy')

    data_generator_top = ImageDataGenerator(rescale=1. / 255)

    # Make top model generator for training set
    generator_top_train = data_generator_top.flow_from_directory(train_data_path,
                                                                 target_size=(img_width,img_height),
                                                                 batch_size=batch_size,
                                                                 class_mode='categorical',
                                                                 shuffle=False)

    nb_train_samples = len(generator_top_train.filenames)
    num_classes = len(generator_top_train.class_indices)

    # We save these to use in prediction step
    np.save('class_indices.npy', generator_top_train.class_indices)

    # Get the category labels for the training data
    train_labels = generator_top_train.classes

    train_labels = to_categorical(train_labels, num_classes=num_classes)

    # Make top model generator for validation set
    validation_data = np.load('bottleneck_features_validation.npy')

    generator_top_validation = data_generator_top.flow_from_directory(validation_data_path,
                                                                 target_size=(img_width,img_height),
                                                                 batch_size=batch_size,
                                                                 class_mode=None,
                                                                 shuffle=False)

    nb_validation_samples = len(generator_top_validation.filenames)

    # Get the category labels for the validation data
    validation_labels = generator_top_validation.classes

    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    # Lets make the model shall we?
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(validation_data,validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("Accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("Loss: {}".format(eval_loss))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

if __name__ == '__main__':
    make_save_bottleneck_features()
    train_top_model()
