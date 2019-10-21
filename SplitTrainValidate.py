import glob, shutil, os
from random import shuffle

# Size of the input image.
INPUT_SIZE = 250

# Assume that all images are in train folder and copy
#  given fraction of random image to validation folder.
VALIDATION_RATIO = 0.25

catFiles    = glob.glob("train_{}/cat/*".format(INPUT_SIZE))
dogFiles    = glob.glob("train_{}/dog/*".format(INPUT_SIZE))
numCatFiles = len(catFiles)
numDogFiles = len(dogFiles)

print("Number of training images for cat: {}".format(numCatFiles))
print("Number of training images for dog: {}".format(numDogFiles))

if numCatFiles != numDogFiles:
	print("Number of images for cats and dogs must be same.")
	exit(1)

shuffle(catFiles)
shuffle(dogFiles)

os.makedirs("validate_{}/cat".format(INPUT_SIZE), exist_ok=False)
os.makedirs("validate_{}/dog".format(INPUT_SIZE), exist_ok=False)

numValidationImages = int(numCatFiles * VALIDATION_RATIO)
print("Moving {} images from train_{} to validate_{}".format(numValidationImages*2, INPUT_SIZE, INPUT_SIZE))
for i in range(numValidationImages):
	catFileName = catFiles[i].replace("train", "validate")
	dogFileName = dogFiles[i].replace("train", "validate")
	shutil.move(catFiles[i], catFileName)
	shutil.move(dogFiles[i], dogFileName)