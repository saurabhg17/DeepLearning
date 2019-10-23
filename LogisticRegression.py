import argparse, glob, os, sys
import numpy as np
from random import shuffle
from PIL import Image
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

DEFAULT_IMG_SIZE = 100

def main():
	parser = argparse.ArgumentParser(description="Train a image classifier using Logistic Regression.")
	parser.add_argument("--imageSize" , type=int, help="The image size to use", default=DEFAULT_IMG_SIZE)
	
	parser.add_argument("--rescale"   , dest='rescale', action='store_true' , help="Rescale images between 0 and 1.")
	parser.add_argument("--no-rescale", dest='rescale', action='store_false', help="Don't rescale images.")
	parser.set_defaults(rescale=False)
	
	args = parser.parse_args()
	
	X_train, y_train, X_test, y_test = loadDataSet(args.imageSize, args.rescale)
	print("Size and type of training data  : {} and {}".format(X_train.shape, X_train.dtype))
	print("Size and type of training labels: {} and {}".format(y_train.shape, y_train.dtype))
	print("Size and type of testing  data  : {} and {}".format(X_test.shape, X_test.dtype))
	print("Size and type of testing  labels: {} and {}".format(y_test.shape, y_test.dtype))
	
	print("Fitting logistic classifies on the data...")
	classifier = LogisticRegressionCV(cv=3, n_jobs=8, max_iter=100, verbose=0)
	classifier.fit(X_train, y_train)
	np.save("LogisticRegressionCV_{}".format(dataSetSuffix(args.imageSize, args.rescale)), classifier)
	
	print("Calculating training accuracy...")
	train_acc = classifier.score(X_train, y_train) * 100
	print("Training accuracy: {:.2f}%".format(train_acc))
	
	print("Calculating testing accuracy...")
	test_acc = classifier.score(X_test, y_test) * 100
	print("Testing accuracy: {:.2f}%".format(test_acc))
	
	with open("LogisticRegression.csv", "a") as _file:
		_file.write("{}, {}, {:0.2f}, {:0.2f}\n".format(args.imageSize, args.rescale, train_acc, test_acc))


def loadDataSet(imgSize, rescale):
	dataSetName = "Data" + dataSetSuffix(imgSize, rescale) + ".npz"
	
	if os.path.isfile(dataSetName):
		print("Loading existing dataset")
		Data    = np.load(dataSetName)
		X_train = Data["X_train"]
		y_train = Data["y_train"].flatten()
		X_test  = Data["X_test"]
		y_test  = Data["y_test"].flatten()
	else:
		print("Creating a new  dataset")
		X_train, y_train = loadImages(["data/train/**/*", "data/validate/**/*"], imgSize, rescale)
		X_test , y_test  = loadImages("data/test/**/*", imgSize, rescale)
		y_train = y_train.flatten()
		y_test  = y_test.flatten()
		np.savez(dataSetName, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
	
	return X_train, y_train, X_test, y_test


def dataSetSuffix(imgSize, rescale):
	suffix = "_{}".format(imgSize)
	if rescale:
		suffix = suffix + "_Rescaled"
	return suffix


def loadImages(filePath, imgSize, rescale):
	fileNames = []
	if isinstance(filePath, list):
		for fp in filePath:
			fileNames.extend(glob.glob(fp))
	else:
		fileNames = glob.glob(filePath)
	
	shuffle(fileNames)
	
	numImages = len(fileNames)
	print("Number of images to load: {}".format(numImages))
	
	X = np.zeros((numImages, imgSize*imgSize*3), dtype = np.float32 if rescale else np.int32)
	Y = np.zeros((numImages                , 1), dtype = np.float32 if rescale else np.int32)
	
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
