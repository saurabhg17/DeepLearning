import glob, multiprocessing, sys, os
from timeit import default_timer as timer
from PIL import Image as pil_image

numThreads = multiprocessing.cpu_count()-1
imageSize  = 250

def main():
	allFiles = glob.glob("train_orig/**/*.jpg")
	print("Number of training images: {}".format(len(allFiles)))
	print("")

	os.makedirs("train_{}/cat".format(imageSize), exist_ok=True)
	os.makedirs("train_{}/dog".format(imageSize), exist_ok=True)

	print("Resizing images...")
	start = timer()
	with multiprocessing.Pool(numThreads) as p:
		results = p.map(resizeImage, allFiles)
	end = timer()
	print("Time taken to resize imges = {:.3f}".format(end - start))
	print("\n")


def resizeImage(imagePath):
	image      = pil_image.open(imagePath)
	image      = image.convert('RGB')
	image      = image.resize((imageSize, imageSize), pil_image.BICUBIC)
	head, tail = os.path.split(imagePath)
	head       = head.replace("train_orig", "train_{}".format(imageSize))
	image.save("{}/{}".format(head, tail))


if __name__ == "__main__":
	sys.exit(main())
