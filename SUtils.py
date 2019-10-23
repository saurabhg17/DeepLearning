from tensorflow.keras.models import Model
from matplotlib import pyplot as plt

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
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