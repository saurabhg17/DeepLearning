import argparse, glob, os, sys
import numpy as np
from random import shuffle
from PIL import Image
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from enum import Enum
from timeit import default_timer as timer

class Solver(Enum):
	lbfgs = "lbfgs"
	sag   = "sag"
	saga  = "saga"
	
	# magic methods for argparse compatibility
	def __str__(self):
		return self.name
	
	def __repr__(self):
		return str(self)
	
	@staticmethod
	def argparse(s):
		try:
			return Solver[s]
		except KeyError:
			return s


DEFAULT_IMG_SIZE = 100
DEFAULT_SOLVER   = Solver.lbfgs


def main():
	parser = argparse.ArgumentParser(description="Train a image classifier using Logistic Regression.")
	parser.add_argument("--imageSize", type=int            , help="The image size to use"         , default=DEFAULT_IMG_SIZE)
	parser.add_argument("--solver"   , type=Solver.argparse, help="This to use for optimization"  , default=DEFAULT_SOLVER, choices=list(Solver))
	
	parser.add_argument("--rescale"   , dest='rescale', action='store_true' , help="Rescale images between 0 and 1.")
	parser.add_argument("--no-rescale", dest='rescale', action='store_false', help="Don't rescale images.")
	parser.set_defaults(rescale=False)
	
	parser.add_argument("--debug"   , dest='debug', action='store_true' , help="Enable debugging (uses only 100 training examples).")
	parser.add_argument("--no-debug", dest='debug', action='store_false', help="Disable debugging.")
	parser.set_defaults(debug=False)
	
	args = parser.parse_args()
	
	start1 = timer()
	X_train, y_train, X_test, y_test = loadDataSet(args.imageSize, args.rescale)
	
	if args.debug:
		X_train = X_train[0:100, :]
		y_train = y_train[0:100]
		X_test  = X_test [0:100, :]
		y_test  = y_test [0:100]
	
	print("Size and type of training data  : {} and {}".format(X_train.shape, X_train.dtype))
	print("Size and type of training labels: {} and {}".format(y_train.shape, y_train.dtype))
	print("Size and type of testing  data  : {} and {}".format(X_test.shape, X_test.dtype))
	print("Size and type of testing  labels: {} and {}".format(y_test.shape, y_test.dtype))
	end1 = timer()
	timeToLoad = end1 - start1
	print("Time taken to load model = {:.1f} seconds".format(timeToLoad))
	print("")
	
	start2 = timer()
	print("Fitting logistic classifier on the data using {} ...".format(args.solver))
	classifier = LogisticRegressionCV(cv=3, n_jobs=8, max_iter=100, verbose=0, solver=args.solver.value)
	classifier.fit(X_train, y_train)
	np.save("LRModel_{}_{}_{}".format(args.imageSize, "Rescaled" if args.rescale else "", args.solver), classifier)
	end2 = timer()
	timeToFit = end2 - start2
	print("Time taken to fit model = {:.1f} seconds".format(timeToFit))
	print("")
	
	print("Calculating training accuracy...")
	train_acc = classifier.score(X_train, y_train) * 100
	print("Training accuracy: {:.2f}%".format(train_acc))
	print("")
	
	print("Calculating testing accuracy...")
	test_acc = classifier.score(X_test, y_test) * 100
	print("Testing accuracy: {:.2f}%".format(test_acc))
	print("")
	
	end3 = timer()
	totalTime = end3 - start1
	print("Total time taken for logistic regression = {:.1f} seconds".format(totalTime))
	print("")
	
	if not os.path.isfile("LogisticRegression.csv"):
		with open("LogisticRegression.csv", "w") as _file:
			_file.write("Solver, ImageSize, Rescaled, TrainAcc, ValidateAcc, TimeToFit, TotalTime\n")
	
	with open("LogisticRegression.csv", "a") as _file:
		_file.write("{}, {}, {}, {:0.3f}, {:0.3f}, {:0.2f}, {:0.2f}\n".format(args.solver.value, args.imageSize, args.rescale, train_acc, test_acc, timeToFit, totalTime))


def loadDataSet(imgSize, rescale):
	dataSetFileName = "DogsCats_{}{}.npz".format(imgSize, "_Rescaled" if rescale else "")
	
	if os.path.isfile(dataSetFileName):
		print("Loading existing dataset")
		Data    = np.load(dataSetFileName)
		X_train = Data["X_train"]
		y_train = Data["y_train"].flatten()
		X_test  = Data["X_test"]
		y_test  = Data["y_test"].flatten()
	else:
		print("Creating a new  dataset")
		X_train, y_train = loadImages(["data/train/**/*"]  , imgSize, rescale, "training")
		print("")
		X_test , y_test  = loadImages("data/validate/**/*", imgSize, rescale, "testing")
		y_train = y_train.flatten()
		y_test  = y_test.flatten()
		
		if rescale:
			print("")
			print("Rescaling images...")
			print("Sum of feature-wise mean before rescaling: {}".format(np.mean(X_train, axis=0).sum()))
			print("Sum of feature-wise std  before rescaling: {}".format(np.std(X_train, axis=0).sum()))
			
			scaler = preprocessing.StandardScaler().fit(X_train)
			X_train = scaler.transform(X_train)
			X_test  = scaler.transform(X_test)
			
			# Make sure processing is correct.
			mean = abs(np.mean(X_train, axis=0).sum()/X_train.shape[1])
			std  = round(np.std(X_train, axis=0).sum()/X_train.shape[1])
			print("Fature-wise mean after rescaling: {}".format(mean))
			print("Feature-wise std  after rescaling: {}".format(std))
			if mean>1e-6 or std!=1.0:
				print("Error normalizing images")
				sys.exit(1)
			print("")
		
		np.savez(dataSetFileName, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
	
	return X_train, y_train, X_test, y_test


def loadImages(filePath, imgSize, rescale, label):
	fileNames = []
	if isinstance(filePath, list):
		for fp in filePath:
			fileNames.extend(glob.glob(fp))
	else:
		fileNames = glob.glob(filePath)
	
	shuffle(fileNames)
	
	numImages = len(fileNames)
	print("Number of {} images to load: {}".format(numImages, label))
	
	X = np.zeros((numImages, imgSize*imgSize*3), dtype = np.float32 if rescale else np.int32)
	Y = np.zeros((numImages,                 1), dtype = np.float32 if rescale else np.int32)
	
	for i in range(0, numImages):
		image = Image.open(fileNames[i])
		image = image.resize((imgSize, imgSize), resample=Image.LANCZOS)
		
		image = image.convert("RGB")
		if rescale:
			image = np.asarray(image, dtype=np.float32) / 255.0
		else:
			image = np.asarray(image, dtype=np.int32)
		
		red   = image[:, :, 0].reshape(-1)
		green = image[:, :, 1].reshape(-1)
		blue  = image[:, :, 2].reshape(-1)
		image = np.vstack((red, green, blue)).reshape((-1,), order="F") # Interleave three channels.
		
		X[i, :] = image
		
		if "dog" in fileNames[i].lower():
			Y[i, 0] = 1
		elif "cat" in fileNames[i].lower() :
			Y[i, 0] = 0
		else:
			print("Unknown label")
			sys.exit(1)
		
		if i%1000 == 0:
			print("Processed {} of {}".format(i, numImages))
	
	return X, Y


if __name__ == "__main__":
	sys.exit(main())
