import tensorflow as tf
Sequential         = tf.keras.models.Sequential
Dense              = tf.keras.layers.Dense
Dropout            = tf.keras.layers.Dropout
Flatten            = tf.keras.layers.Flatten
Conv2D             = tf.keras.layers.Conv2D
MaxPooling2D       = tf.keras.layers.MaxPooling2D
SGD                = tf.keras.optimizers.SGD
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

from matplotlib import pyplot as plt
import sys



INPUT_SIZE    = 150
NUM_EPOCHS    = 50
LEARNING_RATE = 0.01
SGD_MOMENTUM  = 0.9



def main():
	model = DefineCnnModel()
	

def DefineCnnModel():
	model = Sequential()

	# Block 1.
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Block 2.
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Block 3.
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Output layer.
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(  1, activation='softmax'))
	
	# Define optimization function.
	decay = LEARNING_RATE/NUM_EPOCHS
	sgd   = SGD(lr=LEARNING_RATE, momentum=SGD_MOMENTUM, decay=decay, nesterov=False)
	
	# Compile and return the model.
	model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
	print("\n")
	model.summary()
	print("\n")
	return model


if __name__ == "__main__":
	sys.exit(main())
