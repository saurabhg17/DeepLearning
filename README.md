# Exploring Deep Learning in Python
This repository contains code I used for learning and exploring Deep Learning using Python. 

## TrainCNN.py
I wrote `TrainCNN.py` so that I can easily experiment with some of the important hyperparameters involved in training Deep Convolutional Neural Networks. Fixing hyperparameters in the code makes it difficult to try out different values of a parameter as well as to keep track of the results. `TrainCNN.py` accepts values of the hyperparameters on the command line and write the results to CSV files. One CSV file is generated with detailed information about each run of the program. It contains the values of all hyperparameters and the status of each epoch. Some of the hyperparameters, training loss and error, and validation loss and error is appended to a common CSV file. This CSV file allows easy comparison of the effect of different values of hyperparameters in a single place. Additionally, a chart illustrating training and validation loss, and training and validation error is generated using Matplotlib.

`TrainCNN.py` accepts the following command line arguments:
General Parameters:

`--cnnArch {Custom,VGG16}         The CNN architecture (default: Custom)`

`--classMode {Categorical,Binary} The class mode of the data (default: Categorical)`

Hyper Parameters:

`--optimizer {Adam,SGD,RMSProp} The optimization function (default: Adam)`

`--learningRate LEARNINGRATE    The learning rate (default: 0.0001)`

`--imageSize IMAGESIZE          The image size (default: 224)`

`--numEpochs NUMEPOCHS          The maximum number of epochs to train (default: 30)`

`--batchSize BATCHSIZE          The batch size to use for training (default: 25)`

Output Parameters:

`--outputFileNamePrefix OUTPUTFILENAMEPREFIX The prefix for all output files (default: Foo)`

`--resultsFileName RESULTSFILENAME           File name of the common output CSV (default: Results.csv)`

Regularization Parameters:

`--dropout                      Enable dropout regularization. (default: False)`

`--no-dropout                   Disable dropout regularization. (default: False)`

`--augmentation                 Enable image augmentations. (default: False)`

`--no-augmentation              Disable image augmentations. (default: False)`

`--augMultiplier AUGMULTIPLIER  With image augmentation, number of images in the dataset times this multiplier is used for training. (default: 3)`

Other Parameters:

`--debug     Debugging mode uses a smaller dataset for faster execution. (default: False)`

`--no-debug  Disable debugging. (default: False)`

Most of the parameters are self explanatory, here I explain only some of the more obscure of them:
* `-–cnnArch`: This parameter defines the architecture of the CNN. Two support values are `Custom` and `VGG16`. `Custom` defines a CNN model with four blocks of 2D convolutional and max pooling layers and ends with two fully dense layers. `VGG16` uses imagenet weights with VGG16 model and without the top layer. I added a single dense layer at the end. Note that when using `VGG16` model, image size is fixed at 224 x 244 and `--imageSize` is ignored.
* `--augMultiplier`: When image augmentation is used, number of training steps is scaled with `–augMultiplier` . This essentially increased the number of images with which the CNN is trained.

### Dataset
`TrainCNN.py` expects data to be in the following layout:
* There should be four folder: data/train. data/train_debug, data/validate, and data/validate_debug in the same folder as `TrainCNN.py`.
* For each of these folders, there should be a sub-folder for each class in the dataset.
*_debug* folders are optional and are used with `–debug` switch. The idea is to test the entire code using a smaller dataset which will take less time to execute to wrinkle out any issues or bugs.

### Planned Updates
* Support for different initializers for weights. Currently `he_uniform` is used.
* Support for other pretrained models available in tf.keras.
* Support for early stopping.
* Support for training on multiple GPUs.

### Using TrainCNN.py
While `TrainCNN.py` can be used by itself, it is primarily designed to be used through Windows batch file or Linux shell scripts. For example, to study the effect of batch size on validation loss we can run the following commands sequentially:

Train.cnn.py -–batchSize 16 –outputFileNamePrefix Model_BS16
Train.cnn.py -–batchSize 32 –outputFileNamePrefix Model_BS32
Train.cnn.py -–batchSize 64 –outputFileNamePrefix Model_BS64
Train.cnn.py -–batchSize 128 –outputFileNamePrefix Model_BS128

Running these four command will generate the following files:
* Model_BS16.h5, Model_BS16.csv, and Model_BS16.png.
* Similarly for other three commands.
* Results.csv
