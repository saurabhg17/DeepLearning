import os
# 0 | DEBUG   | [Default] Print all messages
# 1 | INFO    | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR   | Filter out all messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Prints only the error messages.

# Import tensor flow and define a name for all the classes we want to use.
import tensorflow as tf
Sequential         = tf.keras.models.Sequential
Conv2D             = tf.keras.layers.Conv2D
MaxPooling2D       = tf.keras.layers.MaxPooling2D
Flatten            = tf.keras.layers.Flatten
Dense              = tf.keras.layers.Dense
Dropout            = tf.keras.layers.Dropout
Adam               = tf.keras.optimizers.Adam
RMSprop            = tf.keras.optimizers.RMSprop
SGD                = tf.keras.optimizers.SGD
EarlyStopping      = tf.keras.callbacks.EarlyStopping
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# Import other packages.
from matplotlib import pyplot as plt
import argparse, math, os, sys, SUtils
from pprint import pprint


# Define global variables.
TRAIN_DIR                 = "data/train"
VALIDATION_DIR            = "data/validate"
TRAIN_DIR_DEBUG           = "data/train_debug"
VALIDATION_DIR_DEBUG      = "data/validate_debug"
DEFAULT_LEARNING_RATE     = 1e-4
DEFAULT_IMAGE_SIZE        = 224
DEFAULT_OPTIMIZER         = "adam"
DEFAULT_BATCH_SIZE        = 25
DEFAULT_NUM_EPOCHS        = 30
DEFAULT_AUG_MULTIPLIER    = 3
DEFAULT_OUTPUTFILE_PREFIX = "Foo"
DEFAULT_RESULTS_FILENAME  = "Results.csv"


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def getArgs():
	parser = argparse.ArgumentParser(description="Train a CNN model for dog-vs-cat dataset from Kaggle.")
	parser.add_argument("--learningRate"        , "-lr" , type=float, help="The leaning rate to use"              , default=DEFAULT_LEARNING_RATE)
	parser.add_argument("--imageSize"           , "-is" , type=int  , help="The image size to use"                , default=DEFAULT_IMAGE_SIZE)
	parser.add_argument("--optimizer"           , "-op" , type=str  , help="The optimization function to use"     , default=DEFAULT_OPTIMIZER)
	parser.add_argument("--batchSize"           , "-bs" , type=int  , help="The batch size"                       , default=DEFAULT_BATCH_SIZE)
	parser.add_argument("--numEpochs"           , "-ne" , type=int  , help="The number of epochs"                 , default=DEFAULT_NUM_EPOCHS)
	parser.add_argument("--augMultiplier"       , "-am" , type=int  , help="The multiplier for image augmentation", default=DEFAULT_AUG_MULTIPLIER)
	parser.add_argument("--outputFileNamePrefix", "-pre", type=str  , help="The prefix for the output files"      , default=DEFAULT_OUTPUTFILE_PREFIX)
	parser.add_argument("--resultsFileName"     , "-res", type=str  , help="The output CSV file name"             , default=DEFAULT_RESULTS_FILENAME)
	
	parser.add_argument("--dropout"   ,  dest='addDropout', action='store_true' , help="Enable dropout regularization.")
	parser.add_argument("--no-dropout",  dest='addDropout', action='store_false', help="Disable dropout regularization.")
	parser.set_defaults(addDropout=False)
	
	parser.add_argument("--augmentation"   , dest='addAugmentation', action='store_true' , help="Enable image augmentations.")
	parser.add_argument("--no-augmentation", dest='addAugmentation', action='store_false', help="Disable image augmentations.")
	parser.set_defaults(addAugmentation=False)
	
	parser.add_argument("--debug"   , dest='debug', action='store_true' , help="Debugging mode uses a smaller dataset for faster execution.")
	parser.add_argument("--no-debug", dest='debug', action='store_false', help="Disable debugging.")
	parser.set_defaults(debug=False)
	
	args = parser.parse_args()
	
	SUtils.PrintArgs(args, "Command line arguments:")
	print("\n")
	return args
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def main():
	print("\n")
	
	args = getArgs()
	
	
	# Create training and validation directory iterators.
	if args.debug:
		trainDir      = TRAIN_DIR_DEBUG
		validationDir = VALIDATION_DIR_DEBUG
	else:
		trainDir      = TRAIN_DIR
		validationDir = VALIDATION_DIR
	
	print("Creating training and validation image iterators")
	trainIterator, validationIterator = CreateDataIterators(args.imageSize, args.batchSize, args.addAugmentation, trainDir, validationDir)
	print("\n")
	
	
	# Define CNN model.
	model = DefineCnnModel(args.optimizer, args.learningRate, args.imageSize, args.addDropout)
	print("\n")
	
	
	# Fit a model to data.
	earlyStopping   = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=1, mode='auto')
	trainingSteps   = math.ceil(trainIterator.n      / args.batchSize)
	validationSteps = math.ceil(validationIterator.n / args.batchSize)
	if args.addAugmentation:
		trainingSteps = trainingSteps * args.augMultiplier
	
	print("Training CNN model with the following parameters:")
	print("Number of training images         : {}".format(trainIterator.n))
	print("Number of validation images       : {}".format(validationIterator.n))
	print("Number of epochs                  : {}".format(args.numEpochs))
	print("Batch size                        : {}".format(args.batchSize))
	print("Training steps per epoch          : {}".format(trainingSteps))
	print("Validation steps per epoch        : {}".format(validationSteps))
	print("Number of  image used for training: {}".format(trainingSteps * args.batchSize))
	print("Dropout layers                    : {}".format(args.addDropout))
	print("Image augmentation                : {}".format(args.addAugmentation))
	print("")
	
	trainingParameters = {
		"NumTrainImages"          : trainIterator.n,
		"NumValidationImages"     : validationIterator.n,
		"TrainingStepsPerEpoch"   : trainingSteps,
		"ValidationStepsPerEpoch" : validationSteps,
		"NumImagesUsedForTraining": trainingSteps * args.batchSize,
	}
	
	history = model.fit_generator(trainIterator, 
	                              steps_per_epoch  = trainingSteps, 
								  epochs           = args.numEpochs, 
								  validation_data  = validationIterator, 
								  validation_steps = validationSteps,
								  callbacks        = [earlyStopping],
								  verbose          = 1)
	model.save(args.outputFileNamePrefix + ".h5")
	print("\n")
	
	# Save results.
	SUtils.GenerateLossAndAccuracyChart(history, args.outputFileNamePrefix + ".png")
	SaveFinalResults(history, args)
	SaveHistory(history, trainingParameters, args)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def DefineCnnModel(optimizerName, learningRate, imageSize, addDropout):
	model = Sequential()
	
	# Block 1.
	model.add(Conv2D( 32, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform", input_shape=(imageSize, imageSize, 3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Block 2.
	model.add(Conv2D( 64, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
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
	#model.add(Conv2D(512, (1, 1), padding="same", activation="relu", kernel_initializer="he_uniform"))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#if addDropout:
	#	model.add(Dropout(0.25))
	
	# Output layer.
	model.add(Flatten())
	if addDropout:
		model.add(Dropout(0.5))
	model.add(Dense(512, activation="relu", kernel_initializer="he_uniform"))
	model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
	model.add(Dense(  2, activation="softmax"))
	
	# define optimizer
	optimizer = None
	if optimizerName.lower() == "adam":
		optimizer = Adam(lr=learningRate)
	elif optimizerName.lower() == "rmsprop":
		optimizer = RMSprop(lr=learningRate)
	elif optimizerName.lower() == "sgd":
		optimizer = SGD(lr=learningRate)
	else:
		print("{} is not a valid optimizer name.".format(optimizerName))
		sys.exit(1)
	
	# Compile and return the model.
	model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
	model.summary()
	return model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def CreateDataIterators(imageSize, batchSize, addAugmentation, trainDir, validationDir):
	# Rescale pixel values to be between 0.0 and 1.0
	if addAugmentation:
		trainDatagen = ImageDataGenerator(rescale            = 1.0/255.0,
                                          rotation_range     = 40,
										  zoom_range         = 0.2,
										  horizontal_flip    = True,
										  vertical_flip      = False,
										  shear_range        = 0.2,
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
	validationDatagen  = ImageDataGenerator(rescale= 1.0/255.0)
	validationIterator = validationDatagen.flow_from_directory(validationDir,
	                                                           target_size   = (imageSize, imageSize),
															   color_mode    = "rgb",
															   interpolation = "bicubic",
															   batch_size    = batchSize,
															   class_mode    = "categorical",
															   shuffle       = True)
	
	return trainIterator, validationIterator
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #	


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def SaveFinalResults(history, args):
	N                   = len(history.history["val_loss"])
	train_loss          = history.history["loss"][N-1]
	train_accuracy      = history.history["accuracy"][N-1] * 100.0
	validation_loss     = history.history["val_loss"][N-1]
	validation_accuracy = history.history["val_accuracy"][N-1] * 100.0
	
	if not os.path.isfile(args.resultsFileName):
		with open(args.resultsFileName, "w") as _file:
			_file.write("Optimizer, LearningRate, ImageSize, BatchSize, NumEpoch, Regularized, AugMultiplier, TrainLoss, TrainAcc, ValidLoss, ValidateAcc\n")
	
	with open(args.resultsFileName, "a") as _file:
		_file.write("{}, ".format(args.optimizer))
		_file.write("{}, ".format(args.learningRate))
		_file.write("{}, ".format(args.imageSize))
		_file.write("{}, ".format(args.batchSize))
		_file.write("{}, ".format(args.numEpochs))
		_file.write("{}, ".format(args.addDropout or args.addAugmentation))
		_file.write("{}, ".format(args.augMultiplier))
		_file.write("{:.3f}, ".format(train_loss))
		_file.write("{:.3f}, ".format(train_accuracy))
		_file.write("{:.3f}, ".format(validation_loss))
		_file.write("{:.3f}, ".format(validation_accuracy))
		_file.write("\n")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def SaveHistory(history, trainingParameters, args):
	with open(args.outputFileNamePrefix + ".csv", "w") as _file:
		# Write command line arguments.
		_file.write("Argument, Value\n")
		for arg in vars(args):
			_file.write("{}, {}\n".format(arg, getattr(args, arg)))
		
		# Write training parameters.
		for k, v in trainingParameters.items():
			_file.write("{}, {}\n".format(k, v))
		
		# Write training history.
		_file.write("\n\n")
		_file.write("Epoch, Training Loss, Training Accuracy, Validation Loss, Validation Accuracy\n")
		N = len(history.history["val_loss"])
		for e in range(0, N):
			train_loss          = history.history["loss"][e]
			train_accuracy      = history.history["accuracy"][e] * 100.0
			validation_loss     = history.history["val_loss"][e]
			validation_accuracy = history.history["val_accuracy"][e] * 100.0
			_file.write("{}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(e, train_loss, train_accuracy, validation_loss, validation_accuracy))
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


if __name__ == "__main__":
	sys.exit(main())
