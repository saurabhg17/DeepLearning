import glob, shutil, os
from random import shuffle

# Assume that all images are in train folder and copy given fraction of random image to validation folder.
valRatio = 0.2

catFiles = glob.glob("train/cat*")
dogFiles = glob.glob("train/dog*")
print("Number of training images for cat: {}".format(len(catFiles)))
print("Number of training images for dog: {}".format(len(dogFiles)))

shuffle(catFiles)
shuffle(dogFiles)

numValidationImages = int(len(catFiles) * valRatio)
print("Moving {} images from train to validate".format(numValidationImages*2))
for i in range(numValidationImages):
    catFileName = os.path.join("validate", catFiles[i].replace("train\\", ""))
    dogFileName = os.path.join("validate", dogFiles[i].replace("train\\", ""))
    
    shutil.move(catFiles[i], catFileName)
    shutil.move(dogFiles[i], dogFileName)