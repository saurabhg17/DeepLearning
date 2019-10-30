import os
# 0 | DEBUG   | [Default] Print all messages
# 1 | INFO    | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR   | Filter out all messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Print only the error messages.


# Import tensor flow and define a name for all the classes we want to use.
import tensorflow as tf
Sequential            = tf.keras.models.Sequential
Conv2D                = tf.keras.layers.Conv2D
MaxPooling2D          = tf.keras.layers.MaxPooling2D
Flatten               = tf.keras.layers.Flatten
Dense                 = tf.keras.layers.Dense
Dropout               = tf.keras.layers.Dropout
Adam                  = tf.keras.optimizers.Adam
RMSprop               = tf.keras.optimizers.RMSprop
SGD                   = tf.keras.optimizers.SGD
EarlyStopping         = tf.keras.callbacks.EarlyStopping
ImageDataGenerator    = tf.keras.preprocessing.image.ImageDataGenerator
VGG16                 = tf.keras.applications.vgg16.VGG16
VGG16_PreprocessInput = tf.keras.applications.vgg16.preprocess_input


# Import other packages.
from matplotlib import pyplot as plt
import argparse, math, os, sys, SUtils


# Define global variables.
TRAIN_DIR                 = "data/train"
VALIDATION_DIR            = "data/validate"
TRAIN_DIR_DEBUG           = "data/train_debug"
VALIDATION_DIR_DEBUG      = "data/validate_debug"

DEFAULT_CNN_ARCH          = SUtils.CnnArch.Custom
DEFAULT_CLASS_MODE        = SUtils.ClassMode.Categorical
DEFAULT_OPTIMIZER         = SUtils.Optimizer.Adam
DEFAULT_LEARNING_RATE     = 1e-4
DEFAULT_IMAGE_SIZE        = 224
DEFAULT_BATCH_SIZE        = 25
DEFAULT_NUM_EPOCHS        = 30
DEFAULT_AUG_MULTIPLIER    = 3
DEFAULT_OUTPUTFILE_PREFIX = "Foo"
DEFAULT_RESULTS_FILENAME  = "Results.csv"


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def getArgs():
	parser = argparse.ArgumentParser(description="Train a CNN model for classifying images into classes.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	group = parser.add_argument_group("General Parameters")
	group.add_argument("--cnnArch"  , type=SUtils.CnnArch.argparse  , help="The CNN architecture"      , default=DEFAULT_CNN_ARCH  , choices=list(SUtils.CnnArch))
	group.add_argument("--classMode", type=SUtils.ClassMode.argparse, help="The class mode of the data", default=DEFAULT_CLASS_MODE, choices=list(SUtils.ClassMode))
	
	group = parser.add_argument_group("Hyper Parameters")
	group.add_argument("--optimizer"   , type=SUtils.Optimizer.argparse, help="The optimization function"            , default=DEFAULT_OPTIMIZER, choices=list(SUtils.Optimizer))
	group.add_argument("--learningRate", type=float                    , help="The learning rate"                    , default=DEFAULT_LEARNING_RATE)
	group.add_argument("--imageSize"   , type=int                      , help="The image size"                       , default=DEFAULT_IMAGE_SIZE)
	group.add_argument("--numEpochs"   , type=int                      , help="The maximum number of epochs to train", default=DEFAULT_NUM_EPOCHS)
	group.add_argument("--batchSize"   , type=int                      , help="The batch size to use for training"   , default=DEFAULT_BATCH_SIZE)
	
	group = parser.add_argument_group("Output Parameters")
	group.add_argument("--outputFileNamePrefix", type=str, help="The prefix for all output files"   , default=DEFAULT_OUTPUTFILE_PREFIX)
	group.add_argument("--resultsFileName"     , type=str, help="File name of the common output CSV", default=DEFAULT_RESULTS_FILENAME)
	
	group = parser.add_argument_group("Regularization Parameters")
	group.add_argument("--dropout"   ,  dest='addDropout', action='store_true' , help="Enable dropout regularization.")
	group.add_argument("--no-dropout",  dest='addDropout', action='store_false', help="Disable dropout regularization.")
	group.set_defaults(addDropout=False)
	
	group.add_argument("--augmentation"   , dest='addAugmentation', action='store_true' , help="Enable image augmentations.")
	group.add_argument("--no-augmentation", dest='addAugmentation', action='store_false', help="Disable image augmentations.")
	group.set_defaults(addAugmentation=False)
	group.add_argument("--augMultiplier", type=int, help="With image augmentation, number of images in the dataset times this multiplier is used for training.", default=DEFAULT_AUG_MULTIPLIER)
	
	group = parser.add_argument_group("Other Parameters")
	group.add_argument("--debug"   , dest='debug', action='store_true' , help="Debugging mode uses a smaller dataset for faster execution.")
	group.add_argument("--no-debug", dest='debug', action='store_false', help="Disable debugging.")
	group.set_defaults(debug=False)
	
	args = parser.parse_args()
	
	# Check compatibility of loss function and optimizer.
	if args.classMode == SUtils.ClassMode.Binary and args.optimizer != SUtils.Optimizer.RMSProp:
		print("Binary loss function can only be used with RMSProp optimizer")
		sys.exit(1)
	
	if args.cnnArch == SUtils.CnnArch.VGG16:
		print("Warning: Image size is not used when vgg16 is used.")
	
	SUtils.PrintArgs(args, "Command line arguments:")
	return args
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def main():
	print("")
	args = getArgs()
	print("\n")
	
	# --------------------------------------------------.
	# Create training and validation directory iterators.
	if args.debug:
		trainDir      = TRAIN_DIR_DEBUG
		validationDir = VALIDATION_DIR_DEBUG
	else:
		trainDir      = TRAIN_DIR
		validationDir = VALIDATION_DIR
	
	print("Creating training and validation image iterators")
	trainIterator, validationIterator = CreateDataIterators(args.cnnArch, args.classMode, args.imageSize, args.batchSize, args.addAugmentation, trainDir, validationDir)
	print("\n")
	
	
	# --------------------------------------------------.
	# Define CNN model.
	if args.cnnArch == SUtils.CnnArch.Custom:
		model = DefineCnnModel(args.classMode, args.optimizer, args.learningRate, args.imageSize, args.addDropout)
	elif args.cnnArch == SUtils.CnnArch.VGG16:
		model = DefineVggTopModel(args.classMode, args.optimizer, args.learningRate, args.addDropout)
	print("\n")
	
	
	# --------------------------------------------------.
	# Fit a model to data.
	#earlyStopping   = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=1, mode='auto')
	trainingSteps   = math.ceil(trainIterator.n      / args.batchSize)
	validationSteps = math.ceil(validationIterator.n / args.batchSize)
	if args.addAugmentation:
		trainingSteps = trainingSteps * args.augMultiplier
	
	print("Training CNN model with the following parameters:")
	print("CNN architecture                  : {}".format(args.cnnArch))
	print("Class mode                        : {}".format(args.classMode))
	print("Optimizer                         : {}".format(args.optimizer))
	print("Learning Rate                     : {}".format(args.learningRate))
	print("Image Size                        : {}x{}".format(args.imageSize, args.imageSize))
	print("Number of epochs                  : {}".format(args.numEpochs))
	print("Batch size                        : {}".format(args.batchSize))
	print("Dropout layers                    : {}".format(args.addDropout))
	print("Image augmentation                : {}".format(args.addAugmentation))
	print("Image augmentation multiplier     : {}".format(args.augMultiplier))
	print("Number of training images         : {}".format(trainIterator.n))
	print("Number of validation images       : {}".format(validationIterator.n))
	print("Training steps per epoch          : {}".format(trainingSteps))
	print("Validation steps per epoch        : {}".format(validationSteps))
	print("Number of images used for training: {}".format(trainingSteps * args.batchSize))
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
								  #callbacks        = [earlyStopping],
								  verbose          = 1)
	print("\n")
	
	
	# --------------------------------------------------.
	# Save results.
	model.save(args.outputFileNamePrefix + ".h5")
	SUtils.GenerateLossAndAccuracyChart(history, args.outputFileNamePrefix + ".png")
	SaveStatsToCommanFile(history, args)
	SaveHistory(history, trainingParameters, args)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def DefineCnnModel(classMode, optimizer, learningRate, imageSize, addDropout):
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
	
	lossFx = classMode.value.lower() + "_crossentropy"
	if classMode == SUtils.ClassMode.Categorical:
		model.add(Dense(2, activation="softmax"))
	elif classMode == SUtils.ClassMode.Binary:
		model.add(Dense(1, activation="sigmoid"))
	
	# define optimizer
	opt = None
	if optimizer == SUtils.Optimizer.Adam:
		opt = Adam(lr=learningRate)
	elif optimizer == SUtils.Optimizer.RMSProp:
		opt = RMSprop(lr=learningRate)
	elif optimizer == SUtils.Optimizer.SGD:
		opt = SGD(lr=learningRate)
	
	# Compile and return the model.
	model.compile(optimizer=opt, loss=lossFx, metrics=["accuracy"])
	model.summary()
	return model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #	
def DefineVggTopModel(classMode, optimizer, learningRate, addDropout):
	vgg16Base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
	vgg16Base.trainable = False
	
	model = Sequential()
	model.add(vgg16Base)
	
	model.add(Flatten())
	if addDropout:
		model.add(Dropout(0.5))
	
	model.add(Dense(512, activation='relu'))
	if addDropout:
		model.add(Dropout(0.5))
	
	model.add(Dense(256, activation='relu'))
	
	# Compile and return the model.
	lossFx = classMode.value.lower() + "_crossentropy"
	if classMode == SUtils.ClassMode.Categorical:
		model.add(Dense(2, activation="softmax"))
	elif classMode == SUtils.ClassMode.Binary:
		model.add(Dense(1, activation="sigmoid"))
	
	# Define optimizer
	opt = None
	if optimizer == SUtils.Optimizer.Adam:
		opt = Adam(lr=learningRate)
	elif optimizer == SUtils.Optimizer.RMSProp:
		opt = RMSprop(lr=learningRate)
	elif optimizer == SUtils.Optimizer.SGD:
		opt = SGD(lr=learningRate)
	
	# Compile and return the model.
	model.compile(optimizer=opt, loss=lossFx, metrics=["accuracy"])
	model.summary()
	return model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #	



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def CreateDataIterators(cnnArch, classMode, imageSize, batchSize, addAugmentation, trainDir, validationDir):
	if not addAugmentation:
		trainDatagen  = ImageDataGenerator(rescale = 1.0/255.0)
	elif addAugmentation and cnnArch == SUtils.CnnArch.Custom:
		trainDatagen = ImageDataGenerator(rescale            = 1.0/255.0,
                                          rotation_range     = 40,
										  zoom_range         = 0.2,
										  horizontal_flip    = True,
										  vertical_flip      = False,
										  shear_range        = 0.2,
										  width_shift_range  = 0.2,
										  height_shift_range = 0.2)
	elif addAugmentation and cnnArch == SUtils.CnnArch.VGG16:
		trainDatagen = ImageDataGenerator(preprocessing_function=VGG16_PreprocessInput,
                                          rotation_range     = 40,
										  zoom_range         = 0.2,
										  horizontal_flip    = True,
										  vertical_flip      = False,
										  shear_range        = 0.2,
										  width_shift_range  = 0.2,
										  height_shift_range = 0.2)
	
	trainIterator = trainDatagen.flow_from_directory(trainDir,
	                                                 target_size   = (imageSize, imageSize),
													 color_mode    = "rgb",
													 interpolation = "bicubic",
													 batch_size    = batchSize,
													 class_mode    = classMode.value.lower(),
													 shuffle       = True)
	
	if cnnArch == SUtils.CnnArch.Custom:
		validationDatagen  = ImageDataGenerator(rescale= 1.0/255.0)
	elif cnnArch == SUtils.CnnArch.VGG16:
		validationDatagen  = ImageDataGenerator(preprocessing_function=VGG16_PreprocessInput)
	
	validationIterator = validationDatagen.flow_from_directory(validationDir,
	                                                           target_size   = (imageSize, imageSize),
															   color_mode    = "rgb",
															   interpolation = "bicubic",
															   batch_size    = batchSize,
															   class_mode    = classMode.value.lower(),
															   shuffle       = True)
	
	return trainIterator, validationIterator
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #	


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def SaveStatsToCommanFile(history, args):
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
