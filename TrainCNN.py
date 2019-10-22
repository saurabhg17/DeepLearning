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
Dropout            = tf.keras.layers.Dropout
SGD                = tf.keras.optimizers.SGD
EarlyStopping      = tf.keras.callbacks.EarlyStopping
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

from matplotlib import pyplot as plt
import argparse, sys

TRAIN_DIR            = "data/train"
VALIDATION_DIR       = "data/validate"
TRAIN_DIR_DEBUG      = "data/train_debug"
VALIDATION_DIR_DEBUG = "data/validate_debug"
DEFAULT_IMAGE_SIZE   = 200
DEFAULT_BATCH_SIZE   = 32
DEFAULT_NUM_EPOCHS   = 25


def main():
	parser = argparse.ArgumentParser(description="Train a CNN model to determine if an image has dog or cat.")
	parser.add_argument("--imageSize"  , "-is", type=int, help="The image size to use"           , default=DEFAULT_IMAGE_SIZE)
	parser.add_argument("--optimizer"  , "-op", type=str, help="The optimization function to use", default="adam")
	parser.add_argument("--batchSize"  , "-bs", type=int, help="The batch size (32, 64, 128)"    , default=DEFAULT_BATCH_SIZE)
	parser.add_argument("--numEpochs"  , "-ne", type=int, help="The number of epochs"            , default=DEFAULT_NUM_EPOCHS)
	parser.add_argument("--outputChart", "-oc", type=str, help="The output chart file name"      , default="Results.png")
	parser.add_argument("--outputCSV"  , "-od", type=str, help="The output CSV file name"        , default="Results.csv")
	
	parser.add_argument("--dropout"   ,  dest='addDropout', action='store_true' , help="Enable dropout regularization.")
	parser.add_argument("--no-dropout",  dest='addDropout', action='store_false', help="Disable dropout regularization.")
	parser.set_defaults(addDropout=False)
	
	parser.add_argument("--augmentation"   , dest='addAugmentation', action='store_true' , help="Enable image augmentations.")
	parser.add_argument("--no-augmentation", dest='addAugmentation', action='store_false', help="Disable image augmentations.")
	parser.set_defaults(addAugmentation=False)
	
	parser.add_argument("--debug"   , dest='debug', action='store_true' , help="Enable debugging with a smaller dataset.")
	parser.add_argument("--no-debug", dest='debug', action='store_false', help="Disable debugging.")
	parser.set_defaults(debug=False)
	
	args = parser.parse_args()
	
	print("\n")
	printArgs(args)
	print("\n")
	
	# If debugging use a smaller data set.
	if args.debug:
		trainDir      = TRAIN_DIR_DEBUG
		validationDir = VALIDATION_DIR_DEBUG
	else:
		trainDir      = TRAIN_DIR
		validationDir = VALIDATION_DIR
	
	model = DefineCnnModel(args.optimizer, args.imageSize, args.addDropout)
	print("\n")
	
	print("Creating training and validation image iterators")
	trainIterator, validationIterator = CreateDataIterators(args.imageSize, args.batchSize, args.addAugmentation, trainDir, validationDir)
	print("\n")
	
	print("Training CNN model")
	earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=1, mode='auto')
	history = model.fit_generator(trainIterator, 
	                              steps_per_epoch  = len(trainIterator), 
								  epochs           = args.numEpochs, 
								  verbose          = 1,
								  validation_data  = validationIterator, 
								  validation_steps = len(validationIterator),
								  callbacks        = [earlyStopping])
	print("\n")
	
	GenerateCharts(history, args.outputChart)
	SaveResults(history, args)
	
	
def DefineCnnModel(optimizer, imageSize, addDropout):
	model = Sequential()
	
	# Block 1.
	model.add(Conv2D( 32, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform", input_shape=(imageSize, imageSize, 3)))
	model.add(MaxPooling2D(pool_size=(4, 4)))
	if addDropout:
		model.add(Dropout(0.25))
	
	# Block 2.
	model.add(Conv2D( 64, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	if addDropout:
		model.add(Dropout(0.25))
	
	# Block 3.
	model.add(Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	if addDropout:
		model.add(Dropout(0.25))
	
	# Block 4.
	model.add(Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	if addDropout:
		model.add(Dropout(0.25))
	
	# Block 5.
	model.add(Conv2D(512, (1, 1), padding="same", activation="relu", kernel_initializer="he_uniform"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	if addDropout:
		model.add(Dropout(0.25))
	
	# Output layer.
	model.add(Flatten())
	model.add(Dense(32, activation="relu"))
	if addDropout:
		model.add(Dropout(0.5))
	model.add(Dense( 2, activation="softmax"))
	
	# Compile and return the model.
	model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
	model.summary()
	return model


def CreateDataIterators(imageSize, batchSize, addAugmentation, trainDir, validationDir):
	# Rescale pixel values to be between 0.0 and 1.0
	if addAugmentation:
		trainDatagen = ImageDataGenerator(rescale            = 1.0/255.0,
                                          rotation_range     = 20,
										  zoom_range         = 0.1,
										  brightness_range   = [0.5, 1.5],
										  horizontal_flip    = True,
										  vertical_flip      = False,
										  shear_range        = 0.15,
										  width_shift_range  = 0.2,
										  height_shift_range = 0.2)
	else:
		trainDatagen  = ImageDataGenerator(rescale = 1.0/255.0)
	
	trainIterator = trainDatagen.flow_from_directory(trainDir,
	                                                 target_size   = (imageSize, imageSize),
													 color_mode    = "rgb",
													 interpolation = "bicubic",
													 batch_size    = batchSize,
													 class_mode    = "categorical",
													 shuffle       = True)
	
	# Rescale pixel values to be between 0.0 and 1.0
	if addAugmentation:
		validationDatagen = ImageDataGenerator(rescale            = 1.0/255.0,
                                               rotation_range     = 20,
											   zoom_range         = 0.1,
											   brightness_range   = [0.5, 1.5],
											   horizontal_flip    = True,
											   vertical_flip      = False,
											   shear_range        = 0.15,
											   width_shift_range  = 0.2,
											   height_shift_range = 0.2)
	else:
		validationDatagen = ImageDataGenerator(rescale = 1.0/255.0)
	
	validationIterator = validationDatagen.flow_from_directory(validationDir,
	                                                           target_size   = (imageSize, imageSize),
															   color_mode    = "rgb",
															   interpolation = "bicubic",
															   batch_size    = batchSize,
															   class_mode    = "categorical",
															   shuffle       = True)
	
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
	
	
def printArgs(args):
	L = 0
	for arg in vars(args):
		if len(arg) > L:
			L = len(arg)
	
	print("Training CNN with the following arguments:")
	for arg in vars(args):
		l = len(arg)
		print("{}{}: {}".format(arg, " "*(L-l), getattr(args, arg)))
	
	
if __name__ == "__main__":
	sys.exit(main())
