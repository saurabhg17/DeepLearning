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
from keras_tqdm import TQDMNotebookCallback
import sys



INPUT_SIZE     = 200
NUM_EPOCHS     = 20
LEARNING_RATE  = 0.1
SGD_MOMENTUM   = 0.9
TRAIN_DIR      = "data/train_orig"#.format(INPUT_SIZE)
VALIDATION_DIR = "data/validate_orig"#.format(INPUT_SIZE)
BATCH_SIZE     = 32


def main():
	model = DefineCnnModel()
	trainIterator, validationIterator = CreateDataIterators()
	
	history = model.fit_generator(trainIterator, 
	                              steps_per_epoch=len(trainIterator), 
								  epochs=NUM_EPOCHS, 
								  verbose=2,
								  validation_data=validationIterator, 
								  validation_steps=len(validationIterator))
	
	
def DefineCnnModel():
	model = Sequential()
	
	# Block 1.
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
	model.add(MaxPooling2D(pool_size=(4, 4)))
	
	# Block 2.
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.4))
	
	# Block 3.
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.4))
	
	model.add(Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	#model.add(Dropout(0.4))
	
	model.add(Conv2D(512, (1,1), padding='same', activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	
	# Output layer.
	model.add(Flatten())
	#model.add(Dropout(0.4))
	
	model.add(Dense(32, activation='relu'))
	#model.add(Dropout(0.5))
	
	model.add(Dense( 2, activation='softmax'))
	
	# Define optimization function.
	decay = LEARNING_RATE/NUM_EPOCHS
	sgd   = SGD(lr=LEARNING_RATE, momentum=SGD_MOMENTUM, decay=decay, nesterov=False)
	
	# Compile and return the model.
	model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
	print("\n")
	model.summary()
	print("\n")
	return model


def CreateDataIterators():
	# Rescale pixel values to be between 0.0 and 1.0
	trainDatagen  = ImageDataGenerator(rescale=1.0/255.0)
	trainIterator = trainDatagen.flow_from_directory(TRAIN_DIR,
	                                                 target_size=(INPUT_SIZE, INPUT_SIZE),
													 color_mode="rgb",
													 interpolation="bicubic",
													 batch_size=BATCH_SIZE,
													 class_mode="categorical",
													 shuffle=True)
	
	# Rescale pixel values to be between 0.0 and 1.0
	validationDatagen  = ImageDataGenerator(rescale=1.0/255.0)
	validationIterator = validationDatagen.flow_from_directory(VALIDATION_DIR,
	                                                           target_size=(INPUT_SIZE, INPUT_SIZE),
															   color_mode="rgb",
															   interpolation="bicubic",
															   batch_size=BATCH_SIZE,
															   class_mode='categorical',
															   shuffle=True)
	
	return trainIterator, validationIterator
	

if __name__ == "__main__":
	sys.exit(main())
