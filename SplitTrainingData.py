import glob, shutil, os, sys
from random import shuffle

# Move a number of randomly selected images from train folder 
# to a folder with the give name.
def moveImages(numImagesToMove, newFolderName):
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
	
	os.makedirs("data/{}/cat".format(newFolderName), exist_ok=False)
	os.makedirs("data/{}/dog".format(newFolderName), exist_ok=False)
	
	numImagesToMove = int(numImagesToMove / 2)
	print("Moving {} images from train to {} folder".format(numImagesToMove*2, newFolderName))
	for i in range(numImagesToMove):
		catFileName = catFiles[i].replace("train", newFolderName)
		dogFileName = dogFiles[i].replace("train", newFolderName)
		shutil.move(catFiles[i], catFileName)
		shutil.move(dogFiles[i], dogFileName)

# Assume that all images are in train folder and copy
#  given fraction of randomly selected images to validate and test folders.
NUM_VALIDATION_IMAGES = 25000 * 0.20
NUM_TEST_IMAGES       = 25000 * 0.20

moveImages(NUM_VALIDATION_IMAGES, "validate")
moveImages(NUM_TEST_IMAGES, "test")
