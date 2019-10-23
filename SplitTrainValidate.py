import glob, shutil, os, sys
from random import shuffle

def moveImages(numImagesToMove, newFolder):
	catFiles    = glob.glob("data/train/cat/*")
	dogFiles    = glob.glob("data/train/dog/*")
	numCatFiles = len(catFiles)
	numDogFiles = len(dogFiles)
	
	print("Number of training images for cat: {}".format(numCatFiles))
	print("Number of training images for dog: {}".format(numDogFiles))
	
	if numCatFiles != numDogFiles:
		print("Number of images for cats and dogs must be same.")
		exit(1)
	
	shuffle(catFiles)
	shuffle(dogFiles)
	
	os.makedirs("data/{}/cat".format(newFolder), exist_ok=False)
	os.makedirs("data/{}/dog".format(newFolder), exist_ok=False)
	
	numImagesToMove = int(numImagesToMove / 2)
	print("Moving {} images from train to {} folder".format(numImagesToMove*2, newFolder))
	for i in range(numImagesToMove):
		catFileName = catFiles[i].replace("train", newFolder)
		dogFileName = dogFiles[i].replace("train", newFolder)
		shutil.move(catFiles[i], catFileName)
		shutil.move(dogFiles[i], dogFileName)

# Assume that all images are in train folder and copy
#  given fraction of random images to validation and test folders.
NUM_VALIDATION_IMAGES = 25000 * 0.20
NUM_TEST_IMAGES       = 25000 * 0.20

moveImages(NUM_VALIDATION_IMAGES, "validation")
moveImages(NUM_TEST_IMAGES, "test")
