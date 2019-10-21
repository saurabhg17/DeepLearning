import tensorflow as tf

# Using less than 100% GPU memory prevents "Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED" error.
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#	tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
#	#tf.config.experimental.set_memory_growth(gpus[0], True)

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


TRAIN_DIR          = "data/train_srctest"
VALIDATION_DIR     = "data/validate_srctest"
DEFAULT_IMAGE_SIZE = 200
DEFAULT_BATCH_SIZE = 32
NUM_EPOCHS         = 2


def main():
	parser = argparse.ArgumentParser(description="Train a CNN model to determine if an image has dog or cat.")
	parser.add_argument("--imageSize"  , "-s"  , type=int, help="The image size to use"           , default=DEFAULT_IMAGE_SIZE)
	parser.add_argument("--optimizer"  , "-opt", type=str, help="The optimization function to use", default="adam")
	parser.add_argument("--batchSize"  , "-b"  , type=int, help="The batch size (32, 64, 128)"    , default=DEFAULT_BATCH_SIZE)
	parser.add_argument("--outputChart", "-oi" , type=str, help="The output chart file name"      , default="Results.png")
	parser.add_argument("--outputCSV"  , "-oc" , type=str, help="The output CSV file name"        , default="Results.csv")
	args = parser.parse_args()
	
	model = DefineCnnModel(args.optimizer, args.imageSize)
	trainIterator, validationIterator = CreateDataIterators(args.imageSize, args.batchSize)
	
	print("\n\n")
	history = model.fit_generator(trainIterator, 
	                              steps_per_epoch=len(trainIterator), 
								  epochs=NUM_EPOCHS, 
								  verbose=2,
								  validation_data=validationIterator, 
								  validation_steps=len(validationIterator))
	
	SaveResults(history, args.outputChart)
	
	loss, accuracy = model.evaluate_generator(validationIterator, steps=len(validationIterator), verbose=2)
	N        = len(history.history["val_loss"])
	loss     = history.history["val_loss"][N-1]
	accuracy = history.history["val_accuracy"][N-1]
	with open(args.outputCSV, "a") as _file:
		_file.write("{}, {}, {}, {:.2f}, {:.2f}\n".format(args.optimizer, args.imageSize, args.batchSize, loss, accuracy*100.0))
	
	
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
	print("\n")
	model.summary()
	print("\n")
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
	

def SaveResults(history, fileName):
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


if __name__ == "__main__":
	sys.exit(main())
