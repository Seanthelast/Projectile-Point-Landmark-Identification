## Problem
In North American archaeology, performing a geomorphic analysis on the style and shape of stone projectile points can help to identify the larger cultural trends of past societies. 
One of my friends, an archaeology PhD student in Arizona, wanted to see if deep learning techniques could speed up these analyses by identifying 6 classes of “landmarks” along a certain 
section of a projectile point’s perimeter. I thus set about to try and create a regressive model to do this task, focusing on experimenting with a supervised learning method using a 
regressive CNN model that would be trained on a projectile point dataset. We had done several projects in CS 474 which used CNN models on image data, but all of them were with classification 
problems. I thus decided to research CNN models for regression to get a general idea for my solution. Two projects in particular were of note: one was a digit classification CNN that took in an 
image of a string of numbers and ran it through a CNN with several different fully connected linear layers at the end, each responsible for identifying a specific digit. The other project involved 
using a CNN to process images of corn and return the coordinates for the center of kernels by feeding the CNN output into a sequence of 3 fully connected linear layers ending with only 2 output channels.  
Both of these prior solutions were fairly effective for their tasks and would be incredibly influential to my own experimentation and final solution design. 
## Dataset
My archaeologist friend was able to provide for me a dataset of around 550 images of projectile points from various sites across the Southwest as well as a CSV file for each that provided the 
coordinates of the target landmarks for each. This was a dataset he had made and gave me permission to use. This and related data would be of interest to North American archaeologists, who use projectile 
point data to date sites and learn about exchange patterns. The data did require some preliminary cleaning, with images that were not 1000 x 1000 in size being eliminated and some entries in the CSV file
also being deleted if they did not correspond to an image. This left only 531 images to use. The dataset itself was rather simple, consisting of only 4 columns: the artifact ID-number, the x coordinate, 
the y coordinate, and the number of the class of the particular landmark. The x coordinates existed over a large range, with minimum at 388 and maximum at 971, while the y coordinates having an even greater
range from 2 to 1000, not showing any particular pattern that is easy for one to detect. 
## Approach 
The topology I pursued for my solution was using a CNN approach as this was the primary image processing technique we learned in class and I found it to be the best way to identify features in image data, 
especially when modified for regression. I pursued three primary approaches for CNN models: a model that did only individual classes (which served as a baseline for the other models’ performance), a single 
vector output model that would have only one fully connected head to return all the point coordinates in order of class, and a “multi-headed” model with several “heads” of fully connected linear layers at the
end for each class. Due to my relatively small dataset size, I also decided to try out transfer learning, using pretrained Pytorch models such as Resnet152 by modifying them for regression capabilities, and 
training them on my dataset to compare the results with the models I designed.
	After doing a number of tests with these different types of models, I found that outside of fine-tuning, my multi-headed CNN model gave the best result for multi-class regression. The model itself consisted 
of five Conv2D layers, with the number of channels starting at 3 and increasing gradually up to 128 before being reduced gradually back to 36. Interspersed between these Conv2D layers were 4 ReLU layers, one 
LeakyReLU layer, and two MaxPool2D layers. Large kernel sizes were used to reduce the dimensions of the output image so it could be passed into the 6 “heads”, each consisting of 4 fully connected linear layers
interspersed with ReLU layers and ending with 2 output channels to return coordinate output. The output of each head was then concatenated and returned. In all, the model had 169,545,440 parameters and with 
only 3 epochs took around an hour to train. The Adam optimizer was used, as it had been used for prior class projects and seemed to suit the needs of this model. No pretrained weights were used for this 
particular solution.
The data was split into a train/validation set using an 80/20 split to check for overfitting. The split was accomplished by splitting on the list of artifact ID’s themselves, then when an ID from either
dataset was called for, the image and landmarks corresponding to it were taken from the source files. This model was trained using a training loop that iterated for a certain number of epochs. Prior to the
loop, the data was rescaled to fit a 256x256 format. Within the loop, the input image and its landmarks were grabbed from the data loader, the true coordinates were normalized and the image tensor passed 
into the CNN model. The output tensor of predicted coordinates was then compared to true values using MSE Loss, which was then backpropagated into the model. At the end of each epoch, the average validation
loss was then calculated. In addition to loss, an accuracy function for regression was also developed, with a threshold value being used as a heuristic for determining if a predicted point was in an 
accurate range relative to the actual point. After testing with the single class model, I settled upon a threshold of 0.2. The train and validation losses and accuracies were recorded during the training 
and then graphed to help evaluate the model’s performance. 
## Results
For the best solution, the “multi-headed” CNN, final loss on the train set was recorded at 0.006805636920034885 for the final run, with all of the loss values being under 0.01 in the final epoch of training.
This is comparable to the baseline single class model and the multi-class model using a pretrained Resnet, both of which have loss function results in this range after 3 epochs.  Validation loss was very 
close to the train loss, with each validation loss average value being under 0.01. Train accuracy fluctuated between 0.6 and 1 by the second epoch, with average validation accuracy being close at 0.8 for 
all three epochs, which is also comparable to the baseline and Resnet models. The closeness of the train and validation loss and accuracy values indicates that this model did not overfit on the train data, 
showing that this model generalizes well for this dataset. 
My first algorithm I designed, the single class model solution, was not what I ultimately used as the purpose of this project was to try and design a multi-class regression solution that could match the 
loss and accuracy results of a single class solution.   Based upon the results of my final solution, I would say I did largely solve this problem. I am quite surprised by the accuracy of my results, as 
I thought that the dataset would be too small or just too variable for the CNN model to properly identify. The fact that the results of my multi-headed model even matched the pretrained Resnet model show 
how good this model is and I hope my friend can find much use of it in his work. 

