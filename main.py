# Recreating CNN self, with description and explanations
# Taking help from machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification

# Research Data Augmentation - specifically for images
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import time

# Import Dataset
def load_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # one hot encode labels as we know there are 10 classes
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # View shape
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
    # Images are 32x32 with 3 byte value for colour
    return x_train, y_train, x_test, y_test

#
def prep_images(train, test):
    # convert integers to floats because
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalised images
    train_norm, test_norm = train_norm / 255.0, test_norm / 255.0
    return train_norm, test_norm


# Baseline VGG 3 layer CNN model
# Extreme overfitting as seen in the plot
def define_model():
    model = Sequential()
    # VGG-style architecture for a baseline model
    # VGG 3 - Very Deep Convolutional Networks for Large Scale Image Recognition. This network is characterized by its
    # simplicity, using only 3×3 convolutional layers stacked on top of each other in increasing depth
    # padding ensures height and width of output matches input

    # Conv2D - convolution kernel that is wind with layers input which helps produce a tensor of outputs
    #   https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
    # Conv kernel output filtered image.Trainable elements are the values that compose the kernels. Here 3*3=9 + bias

    # BatchNormalisation - dec training time. reduces the amount by what the hidden unit values shift around (cov shift)
    #   Using batch normalisation leads to using a reduced dropout probability due to added noise & ?

    # MaxPooling2D - Downsamples the i/p representation by taking the max val ovre the window (pool_size) for each dim
    # along features axis. o/p_shape = (i/p_shape - pool_size +1) / strides)
    # Why?

    # Dropout - high predictive capacity of certain neurons regulated using dropout
    #   https://towardsdatascience.com/12-main-dropout-methods-mathematical-and-visual-explanation-58cdc2112293
    # Neurons are omitted in random. In a dense network, for each layer probability p of dropout
    # Hinton recommends 0.2 on i/p layer, 0.5 on hidden layers and 0 for o/p
    # Probabiliy of omission for each neuron follows a Bernoulli dist

    # Flatten - Flattens the i/p. e.g. conv final layer shape (1,10,64) -> flattened shape (640). More

    # Dense - Fully connected layer

    # Feature detection part of model
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    #
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # Example output part of model
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Plot curves of loss and accuracy per epoch
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    plt.savefig('Loss_4' + '_plot_Dropout_BatchNorm.png')
    plt.close()


def run_test_harness():
    # Load data into different variables
    x_train, y_train, x_test, y_test = load_dataset()
    # Normalise images / to_float
    x_train, x_test = prep_images(x_train, x_test)
    # Create model
    model = define_model()
    time_start = time.perf_counter()
    # Fitting model requires params: epochs, batch size (2-32)
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), verbose=0)
    time_elapsed = (time.perf_counter() - time_start)
    # evaluate model
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"time_elapsed: {time_elapsed}")
    print(f"Accuracy: {acc * 100.0}%")

    model.save('final_model.h5')
    # learning curves
    summarize_diagnostics(history)


# Entry point
run_test_harness()
