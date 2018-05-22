import statistics as s

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import copy
from skimage.color import rgb2gray


# Loading the dataset
class DS(Dataset):

    def __init__(self, data_path, label_path, custom):

        X = np.load(data_path)

        X = X.astype(np.float64)
        y = np.load(label_path)
        self.len = X.shape[0]
        result = []

        # First if statement changes images to grasycale + normalization, second statement is only normalization.
        if custom:
            for i in range(self.len):
                img = copy.deepcopy(X[i])
                img = np.transpose(img, (1, 2, 0))
                img_gray = rgb2gray(img).astype(np.float64)

                result.append(img_gray)

            result = np.asarray(result)

            for i in range(self.len):
                np_data = copy.deepcopy(result[i])
                mean = np.mean(np_data).astype(np.float64)  # get the mean and std
                std = (np.std(np_data)).astype(np.float64)

                img__norm = (np_data - mean) / std  # normalizing ndarray

                result[i] = img__norm  # updating the channels

            self.x_data = torch.from_numpy(result).unsqueeze(1).float()
            self.y_data = torch.from_numpy(y).unsqueeze(1).long()

        # This loop prepares data for my the original  CNN.
        # Only normalization is applied
        if custom == False:
            for i in range(self.len):
                np_data = copy.deepcopy(X[i])
                mean = np.mean(np_data).astype(np.float64)  # get the mean and std
                std = (np.std(np_data)).astype(np.float64)
                img_a = np_data[0]  # Extracting single channels from 3 channel image
                img_b = np_data[1]
                img_c = np_data[2]
                img_a_norm = (img_a - mean) / std  # normalizing ndarray
                img_b_norm = (img_b - mean) / std
                img_c_norm = (img_c - mean) / std
                X[i][0] = img_a_norm  # updating the channels
                X[i][1] = img_b_norm
                X[i][2] = img_c_norm

            self.x_data = torch.from_numpy(X).float()
            self.y_data = torch.from_numpy(y).float()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# The feed forward network design
class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size,
                             hidden_size)  # 1st Full-Connected Layer: 3072 (input data) -> 1024 (hidden node)
        self.fc2 = nn.Linear(hidden_size,
                             num_classes)  # 2nd Full-Connected Layer: 1023 (hidden node) -> 10 (output class)

    def forward(self, x):
        out = F.sigmoid(self.fc1(x))  # followed by sigmoid activation function
        out = F.sigmoid(self.fc2(out))

        return out


# The Convolution Neural Network Architecture
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 0)  # First Convolution: 3 input channels with 5x5 Kernal on 6 outputs.
        self.pool = nn.MaxPool2d(2, 2)  # Maxpool operation: Kernel Size 2x2
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)  # First Convolution: 6 input channels with 5x5 Kernal on 16 outputs.
        self.fc1 = nn.Linear(16 * 10 * 10, 120)  # output channel: 120
        self.fc2 = nn.Linear(120, 84)  # output channel: 84
        self.fc3 = nn.Linear(84, 10)  # output class: 10

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = F.relu(self.conv2(out))
        out = out.view(-1, 16 * 10 * 10)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


# The Custom Convolution Neural Network Architecture I designed to improve test accuracy of %46
class Custom_CNN(nn.Module):

    def __init__(self):
        super(Custom_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 0)  # First Convolution: 1 input channel with 5x5 Kernal on 6 outputs.
        self.pool = nn.MaxPool2d(2, 2)  # Maxpool operation: Kernel Size 2x2
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)  # Second Convolution: 6 input channels with 5x5 Kernal on 16 outputs.
        self.conv3 = nn.Conv2d(16, 16, 5, 1, 0)  # Second Convolution: 16 input channels with 5x5 Kernal on 16 outputs.
        # self.softmax = nn.Softmax(dim=10)
        self.fc1 = nn.Linear(16 * 1 * 1, 10)  # output channel: 10
       # self.fc2 = nn.Linear(1, 10)  # output channel: 10

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))  # L0 + L1 + l2
        out = self.pool(F.relu(self.conv2(out)))  # L3 + L4 Now we have 16 feature maps of size 5x5
        out = (F.relu(self.conv3(out)))  # L5 Now we have 16 feature maps of size 1x1
        out = out.view(-1, 16 * 1 * 1)
        out = F.sigmoid((self.fc1(out)))

        return out


# Training and Testing a Feed Forward Neural Network
# Outputs: Training Accuracy of each epoch and Testing Accuracy
def train_and_test_ff_network():
    # Initialize everything to train a feed forward neural network

    # initialize feed forward network
    FFnet = Net(3072, 1024, 10)

    # Specify the training dataset
    dataset = DS('cifar10-data/train_images.npy', 'cifar10-data/train_labels.npy',False)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=64, shuffle=True
                              )

    testset = DS('cifar10-data/test_images.npy', 'cifar10-data/test_labels.npy',False)
    test_loader = DataLoader(dataset=testset,
                             batch_size=64
                             )
    # Specify the loss function
    criterion = nn.CrossEntropyLoss()

    # Specify the optimizer
    optimizer = optim.SGD(FFnet.parameters(), lr=0.001, momentum=0.9)

    # Specify max number of training iterations
    max_epochs = 100

    # Train the feed forward neural network
    epoch_accuracy = np.zeros(max_epochs)

    for epoch in range(max_epochs):
        batch_accuracy = []
        for i, (images, labels) in enumerate(train_loader):  # Load a batch of images with its (index, data, class)

            images = Variable(images.view(images.size(0), -1))  # Convert torch tensor to Variable
            labels = Variable(labels).long()

            optimizer.zero_grad()  # Intialize the hidden weight to all zeros
            y_pred = FFnet(images)  # Forward pass: compute the output class given a image
            loss = criterion(y_pred,
                             labels)  # Compute the loss: difference between the output class and the pre-given label
            loss.backward()  # Backward pass: compute the weight
            optimizer.step()  # Optimizer: update the weights of hidden nodes
            if (i + 1) % 100 == 0:  # Logging
                print('Epoch [%d/%d],Loss: %.4f'
                      % (epoch + 1, max_epochs, loss.data[0]))

            # calculate training accuracy of each epoch

            y_pred_np = y_pred.data.numpy()

            pred_np = np.argmax(y_pred_np, axis=1)

            pred_np = np.reshape(pred_np, len(pred_np), order='F').reshape((len(pred_np), 1))

            label_np = labels.data.numpy().reshape(len(labels), 1)

            correct = 0

            for j in range(y_pred_np.shape[0]):
                if pred_np[j, :] == label_np[j, :]:
                    correct += 1
            batch_accuracy.append(float(correct) / float(len(label_np)))
        epoch_accuracy[epoch] = s.mean(batch_accuracy)

    epoch_number = np.arange(0, max_epochs, 1)

    # Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, epoch_accuracy)
    plt.title('training accuracy over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')
    plt.show()
    correct = 0
    total = 0

    # Calculate testing accuracy
    for images, labels in test_loader:
        images = Variable(images.view(images.size(0), -1))  # Convert torch tensor to Variable
        labels = labels.long()
        outputs = FFnet(images)
        _, predicted = torch.max(outputs.data,
                                 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)  # Increment the total count
        correct += (predicted == labels).sum()  # Increment the correct count
    print('Accuracy of the network on the 2K test images: %d %%' % (100 * correct / total))


# Training and Testing a already given  Convolutional Neural Network
# Outputs: Training Accuracy of each epoch and Testing Accuracy
def train_and_test_convolutional_network():
    # Initialize everything to train a feed forward neural network

    # initialize feed forward network
    ConvNet = CNN()

    # Specify the training dataset
    dataset = DS('cifar10-data/train_images.npy', 'cifar10-data/train_labels.npy', False)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=64, shuffle=True
                              )

    testset = DS('cifar10-data/test_images.npy', 'cifar10-data/test_labels.npy', False)
    test_loader = DataLoader(dataset=testset,
                             batch_size=testset.len
                             )
    # Specify the loss function
    criterion = nn.CrossEntropyLoss()

    # Specify the optimizer
    optimizer = optim.SGD(ConvNet.parameters(), lr=0.001, momentum=0.9)

    # Specify max number of training iterations
    max_epochs = 100

    # Train the feed forward neural network
    epoch_accuracy = np.zeros(max_epochs)
    loss_accuracy = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        batch_accuracy = []
        for i, (images, labels) in enumerate(train_loader):  # Load a batch of images with its (index, data, class)

            images = Variable(images)
            labels = Variable(labels.long())

            optimizer.zero_grad()  # Intialize the hidden weight to all zeros
            y_pred = ConvNet(images)  # Forward pass: compute the output class given a image

            loss = criterion(y_pred, labels)

            # Compute the loss: difference between the output class and the pre-given label
            loss.backward()  # Backward pass: compute the weight
            optimizer.step()  # Optimizer: update the weights of hidden nodes
            if (i + 1) % 100 == 0:  # Logging
                loss_accuracy[epoch] = loss.data[0]
                print('Epoch [%d/%d],Loss: %.4f'
                      % (epoch + 1, max_epochs, loss.data[0]))

            # calculate training accuracy of each epoch

            y_pred_np = y_pred.data.numpy()

            pred_np = np.argmax(y_pred_np, axis=1)

            pred_np = np.reshape(pred_np, len(pred_np), order='F').reshape((len(pred_np), 1))

            label_np = labels.data.numpy().reshape(len(labels), 1)

            correct = 0

            for j in range(y_pred_np.shape[0]):
                if pred_np[j, :] == label_np[j, :]:
                    correct += 1
            batch_accuracy.append(float(correct) / float(len(label_np)))
        epoch_accuracy[epoch] = s.mean(batch_accuracy)

    epoch_number = np.arange(0, max_epochs, 1)
    print(epoch_accuracy[99])

    # Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, epoch_accuracy)
    plt.title('training accuracy over epoches - After Normalization and Customization for grayscale ')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')
    plt.show()

    # Plot the loss data over epoch
    plt.figure()
    plt.plot(epoch_number, loss_accuracy)
    plt.title('Loss over epoches - After Normalization')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss')
    plt.show()

    correct = 0
    total = 0

    # Calculate test accuracy
    for images, labels in test_loader:
        images = Variable(images)  # Convert torch tensor to Variable
        labels = labels.long()
        outputs = ConvNet(images)
        _, predicted = torch.max(outputs.data,
                                 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)  # Increment the total count
        correct += (predicted == labels).sum()  # Increment the correct count
    print('Accuracy of the CNN on the 2000 test images: %d %%' % (100 * correct / total))


# Training and Testing a Custom Designed Convolutional Neural Network
# Outputs: Training Accuracy of each epoch and Testing Accuracy
def train_and_test_custom_convolutional_network():
    # Initialize everything to train a feed forward neural network

    # initialize feed forward network
    ConvNet = Custom_CNN()

    # Specify the training dataset
    # Note that we have to specify third argument of dataset as True. So we can get grayscale image + normalization
    dataset = DS('cifar10-data/train_images.npy', 'cifar10-data/train_labels.npy', False)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=128, shuffle=True
                              )

    testset = DS('cifar10-data/test_images.npy', 'cifar10-data/test_labels.npy', False)
    test_loader = DataLoader(dataset=testset,
                             batch_size=testset.len, shuffle=True
                             )
    # Specify the loss function
    criterion = nn.CrossEntropyLoss()

    # Specify the optimizer
    optimizer = optim.SGD(ConvNet.parameters(), lr=0.003, momentum=0.9)

    # Specify max number of training iterations
    max_epochs = 250

    # Train the feed forward neural network
    epoch_accuracy = np.zeros(max_epochs)
    loss_accuracy = np.zeros(max_epochs)

    for epoch in range(max_epochs):
        batch_accuracy = []
        for i, (inputs, labels) in enumerate(train_loader):  # Load a batch of images with its (index, data, class)

            inputs = Variable(inputs)
            labels = Variable(labels).long()

            optimizer.zero_grad()  # Intialize the hidden weight to all zeros
            outputs = ConvNet(inputs)  # Forward pass: compute the output class given a image

            loss = criterion(outputs, labels)

            # loss = criterion(y_pred,
            #                 labels)  # Compute the loss: difference between the output class and the pre-given label
            loss.backward()  # Backward pass: compute the weight
            optimizer.step()  # Optimizer: update the weights of hidden nodes
            loss_accuracy[epoch] = loss.data[0]
            if (i + 1) % 100 == 0:  # Logging
                loss_accuracy[epoch] = loss.data[0]
                print('Epoch [%d/%d],Loss: %.4f'
                      % (epoch + 1, max_epochs, loss.data[0]))

            # calculate training accuracy of each epoch

            y_pred_np = outputs.data.numpy()

            pred_np = np.argmax(y_pred_np, axis=1)

            pred_np = np.reshape(pred_np, len(pred_np), order='F').reshape((len(pred_np), 1))

            label_np = labels.data.numpy().reshape(len(labels), 1)

            correct = 0

            for j in range(y_pred_np.shape[0]):
                if pred_np[j, :] == label_np[j, :]:
                    correct += 1

            batch_accuracy.append(float(correct) / float(len(label_np)))

        epoch_accuracy[epoch] = s.mean(batch_accuracy)

    epoch_number = np.arange(0, max_epochs, 1)
    print(epoch_accuracy[99])
    # Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, epoch_accuracy)
    plt.title('training accuracy over epoches - for Custom CNN')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')
    plt.show()

    # Plot the loss data over epoch
    plt.figure()
    plt.plot(epoch_number, loss_accuracy)
    plt.title('Loss over epoches - for Custom CNN')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss')
    plt.show()

    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = Variable(test_inputs), Variable(test_labels.long())
        y_pred_test = ConvNet(test_inputs)
        y_pred_test_np = y_pred_test.data.numpy()
        pred_test_np = np.empty(len(y_pred_test_np))
        for k in range(len(y_pred_test_np)):
            pred_test_np[k] = np.argmax(y_pred_test_np[k])

        pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

        label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

        correct_test = 0
        for j in range(y_pred_test_np.shape[0]):
            if pred_test_np[j] == label_test_np[j]:
                correct_test += 1
        print("Test Accuracy: ", (float(correct_test) / float(len(label_test_np))))


#train_and_test_ff_network()
#train_and_test_convolutional_network()
#train_and_test_custom_convolutional_network()
