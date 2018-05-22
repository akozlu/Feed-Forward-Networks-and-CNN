{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf810
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red246\green246\blue239;\red29\green30\blue26;\red244\green0\blue95;
\red152\green224\blue36;\red157\green101\blue255;}
{\*\expandedcolortbl;;\csgenericrgb\c96471\c96471\c93725;\csgenericrgb\c11373\c11765\c10196;\csgenericrgb\c95686\c0\c37255;
\csgenericrgb\c59608\c87843\c14118;\csgenericrgb\c61569\c39608\c100000;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 Experiment 1: \
\
At the end of the script, uncomment the line: \

\b train_and_test_ff_network()
\b0  This function plots training accuracy over each epoch and prints out the test accuracy. \
\
Experiment 2-3: \
\
At the end of the script, uncomment the line: \

\b train_and_test_convolutional_network() 
\b0 This function plots training accuracy over each epoch and prints out the test accuracy. Note that the final version uses normalised images. You can uncomment this under Dataset initialisation function. \
Also note that I gave my dataset class an extra argument called custom. If custom is false, it only normalises the picture. If it is true, it turns the image into grayscale and then normalises it. If you want to train network on grayscale normalized images, make sure custom is True. Also change input channel of self.conv1 to 1 and make sure to change \
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f1 \cf2 \cb3 loss \cf4 = \cf5 criterion\cf2 (y_pred,labels)\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0 \cf0 \cb1 \
to \
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f1 \cf2 \cb3 loss \cf4 = \cf5 criterion\cf2 (outputs, labels.\cf5 squeeze\cf2 (\cf6 1\cf2 ))\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0 \cf0 \cb1 \
Experiment 4: \
These values are not in the script. If you run the script you will get the results I got for experiment 3. The hyper parameters I give in Part 4 need to be coded again. \
\
CUSTOM-CNN: \
If you uncomment the following line, you will run another CNN architecture to train normalised coloured images. It has an extra convolutional layer and different parameters. This is what I used to experiment with a variety of CNN architectures. Initially this was designed to train grayscale images, but I used it for coloured images to show that it was able to reach the same training and testing accuracy with the CNN given in the homework.\
\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f1 \cf5 \cb3 train_and_test_custom_convolutional_network\cf2 ()\
}