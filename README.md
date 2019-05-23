# Feed-Forward-Networks-and-CNN

This project includes implementation of both Feed-forward Neural Network and ConvolutionalNeural Network(CNN) on the CIFAR-10 image dataset. I use Pytorch as the deep learning framework.

Feed Forward Neural Network: 

Architecture: 

| Layer  | Hyperparameters |
| ------------- | ------------- |
| Fully Connected1 | Output channel = 128.  Followed by RandomizedRelu | 
| Fully Connected2 |Output channel = 2.  Followed by RandomizedRelu | 

To run the network uncomment train_and_test_ff_network(). 

The function plots training accuracy over each epoch and prints out the test accuracy. 

Convolutional Neural Network 

Architecture : 

| Layer  | Hyperparameters |
| ------------- | ------------- |
| Convolution1  | Kernel size = (5x5x6), stride = 1, padding = 0.  Followed by ReLU  |
| Pool1 | MaxPool operation.  Kernel size = (2x2) |
| Convolution2 | Kernel size = (5x5x16), stride = 1, padding = 0.  Followed by ReLU | 
| Fully Connected1 | Output channel = 120.  Followed by ReLU | 
| Fully Connected2 | Output channel = 84.  Followed by ReLU | 
| Fully Connected3 | Output channel = 10.  Followed by Sigmoid | 



To run the network uncomment train_and_test_ff_network()  train_and_test_convolutional_network() 
The function plots training accuracy over each epoch and prints out the test accuracy. 


Note that the final version uses normalised images. You can uncomment this under Dataset initialisation function.
Also note that the dataset class has an extra argument called custom. If custom is false, it only normalises the picture. If it is true, it turns the image into grayscale and then normalises it. If you want to train network on grayscale normalized images, make sure custom is True.
You need to make to additional changes. 

Change input channel of self.conv1 to 1.

And change the line loss = criterion(y_pred,labels) to  loss = criterion(outputs, labels.squeeze(1))
