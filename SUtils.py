from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
from enum import Enum

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Define a Enum for class mode to use while training a CNN.
class ClassMode(Enum):
	Categorical = "Categorical"
	Binary      = "Binary"
	
	# magic methods for argparse compatibility
	def __str__(self):
		return self.name
	
	def __repr__(self):
		return str(self)
	
	@staticmethod
	def argparse(s):
		try:
			return ClassMode[s]
		except KeyError:
			return s

class CnnArch(Enum):
	Custom = "Custom"
	VGG16  = "VGG16"
	
	# magic methods for argparse compatibility
	def __str__(self):
		return self.name
	
	def __repr__(self):
		return str(self)
	
	@staticmethod
	def argparse(s):
		try:
			return CnnArch[s]
		except KeyError:
			return s

class Optimizer(Enum):
	Adam    = "Adam"
	SGD     = "SGD"
	RMSProp = "RMSProp"
	
	# magic methods for argparse compatibility
	def __str__(self):
		return self.name
	
	def __repr__(self):
		return str(self)
	
	@staticmethod
	def argparse(s):
		try:
			return Optimizer[s]
		except KeyError:
			return s
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Given a history object returned from the tf.keras.models.Model.fit() or 
# tf.keras.models.Model.fit_genertor() functions, this function generates a loss and accuracy chart
# for both training and validation.
# This chart is very useful to determine if the model under or over fits the data.
def GenerateLossAndAccuracyChart(history, fileName):
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Print the command line arguments passed through argparse package.
def PrintArgs(args, message):
	L = 0
	for arg in vars(args):
		if len(arg) > L:
			L = len(arg)
	
	if message:
		print(message)
	
	for arg in vars(args):
		l = len(arg)
		print("{}{}: {}".format(arg, " "*(L-l), getattr(args, arg)))
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
