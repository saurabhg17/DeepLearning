import os
# 0 | DEBUG   | [Default] Print all messages
# 1 | INFO    | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR   | Filter out all messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Prints only the error messages.

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

Sequential         = tf.keras.models.Sequential
Dense              = tf.keras.layers.Dense
Dropout            = tf.keras.layers.Dropout
Flatten            = tf.keras.layers.Flatten
Conv2D             = tf.keras.layers.Conv2D
MaxPooling2D       = tf.keras.layers.MaxPooling2D
SGD                = tf.keras.optimizers.SGD
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

from matplotlib import pyplot as plt
import argparse, sys

TEST_SRC = False
if not TEST_SRC:
	TRAIN_DIR      = "data/train"
	VALIDATION_DIR = "data/validate"
else:
	TRAIN_DIR      = "data/train_srctest"
	VALIDATION_DIR = "data/validate_srctest"

DEFAULT_IMAGE_SIZE = 200
DEFAULT_BATCH_SIZE = 32
NUM_EPOCHS         = 2


def main():
	parser = argparse.ArgumentParser(description="Train a CNN model to determine if an image has dog or cat.")
	parser.add_argument("--imageSize"  , "-is"  , type=int, help="The image size to use"         , default=DEFAULT_IMAGE_SIZE)
	parser.add_argument("--optimizer"  , "-op", type=str, help="The optimization function to use", default="adam")
	parser.add_argument("--batchSize"  , "-bs"  , type=int, help="The batch size (32, 64, 128)"  , default=DEFAULT_BATCH_SIZE)
	parser.add_argument("--outputChart", "-oc" , type=str, help="The output chart file name"     , default="Results.png")
	parser.add_argument("--outputCSV"  , "-od" , type=str, help="The output CSV file name"       , default="Results.csv")
	args = parser.parse_args()
	
	print("\n")
	model = DefineCnnModel(args.optimizer, args.imageSize)
	print("\n")
	
	print("Creating training and validation image iterators")
	trainIterator, validationIterator = CreateDataIterators(args.imageSize, args.batchSize)
	print("\n")
	
	print("Training CNN model")
	history = model.fit_generator(trainIterator, 
	                              steps_per_epoch=len(trainIterator), 
								  epochs=NUM_EPOCHS, 
								  verbose=1,
								  validation_data=validationIterator, 
								  validation_steps=len(validationIterator))
	print("\n")
	
	GenerateCharts(history, args.outputChart)
	SaveResults(history, args)
	
	
def DefineCnnModel(optimizer, imageSize):
	model = Sequential()
	
	# Block 1.
	model.add(Conv2D( 32, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform", input_shape=(imageSize, imageSize, 3)))
	model.add(MaxPooling2D(pool_size=(4, 4)))
	
	# Block 2.
	model.add(Conv2D( 64, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Block 3.
	model.add(Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Block 4.
	model.add(Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Block 5.
	model.add(Conv2D(512, (1, 1), padding="same", activation="relu", kernel_initializer="he_uniform"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Output layer.
	model.add(Flatten())
	model.add(Dense(32, activation="relu"))
	model.add(Dense( 2, activation="softmax"))
	
	# Compile and return the model.
	model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
	model.summary()
	return model


def CreateDataIterators(imageSize, batchSize):
	# Rescale pixel values to be between 0.0 and 1.0
	trainDatagen  = ImageDataGenerator(rescale=1.0/255.0)
	trainIterator = trainDatagen.flow_from_directory(TRAIN_DIR,
	                                                 target_size=(imageSize, imageSize),
													 color_mode="rgb",
													 interpolation="bicubic",
													 batch_size=batchSize,
													 class_mode="categorical",
													 shuffle=True)
	
	# Rescale pixel values to be between 0.0 and 1.0
	validationDatagen  = ImageDataGenerator(rescale=1.0/255.0)
	validationIterator = validationDatagen.flow_from_directory(VALIDATION_DIR,
	                                                           target_size=(imageSize, imageSize),
															   color_mode="rgb",
															   interpolation="bicubic",
															   batch_size=batchSize,
															   class_mode="categorical",
															   shuffle=True)
	
	return trainIterator, validationIterator
	

def GenerateCharts(history, fileName):
	plt.subplot(211)
	plt.plot(history.history["loss"]    , color="blue"  , label="Train")
	plt.plot(history.history["val_loss"], color="orange", label="Validate")
	plt.title("Cross Entropy Loss")
	plt.legend(loc="upper right")
	
	plt.subplot(212)
	plt.plot(history.history["accuracy"]    , color="blue", label="Train")
	plt.plot(history.history["val_accuracy"], color="orange", label="Validate")
	plt.title("Classification Accuracy")
	plt.legend(loc="lower right")
	
	plt.tight_layout()
	plt.savefig(fileName)
	plt.close()
	
	
def SaveResults(history, args):
	N                   = len(history.history["val_loss"])
	train_loss          = history.history["loss"][N-1]
	train_accuracy      = history.history["accuracy"][N-1] * 100.0
	validation_loss     = history.history["val_loss"][N-1]
	validation_accuracy = history.history["val_accuracy"][N-1] * 100.0
	with open(args.outputCSV, "a") as _file:
		_file.write("{}, {}, {}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n".format(args.optimizer, args.imageSize, args.batchSize, train_loss, train_accuracy, validation_loss, validation_accuracy))
	
	
if __name__ == "__main__":
	sys.exit(main())
