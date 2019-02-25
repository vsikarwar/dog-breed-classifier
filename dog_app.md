
# Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Write your Algorithm
* [Step 6](#step6): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

Make sure that you've downloaded the required human and dog datasets:
* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in this project's home directory, at the location `/dog_images`. 

* Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the home directory, at location `/lfw`.  

*Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.*

In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays `human_files` and `dog_files`.


```python
import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
```

    There are 13233 total human images.
    There are 8351 total dog images.


<a id='step1'></a>
## Step 1: Detect Humans

In this section, we use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  

OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.  In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](output_3_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 

- Human detector, detected 98% humans in human images and 17% human in dog images


```python
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
is_human_detected = 0
for human_img in human_files_short:
    if face_detector(human_img):
        is_human_detected += 1
print("Percentage humans are detected in human images: ", is_human_detected/len(human_files_short))

is_human_detected = 0
for dog_img in dog_files_short:
    if face_detector(dog_img):
        is_human_detected += 1
print("Percentage humans are detected in dog images : ", is_human_detected/len(dog_files_short))
```

    Percentage humans are detected in human images:  0.98
    Percentage humans are detected in dog images :  0.17


We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.


```python
### (Optional) 
### TODO: Test performance of anotherface detection algorithm.
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a [pre-trained model](http://pytorch.org/docs/master/torchvision/models.html) to detect dogs in images.  

### Obtain Pre-trained VGG-16 Model

The code cell below downloads the VGG-16 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  


```python
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
```

    Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.torch/models/vgg16-397923af.pth
    100%|██████████| 553433881/553433881 [00:09<00:00, 57120738.43it/s]


Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.

### (IMPLEMENTATION) Making Predictions with a Pre-trained Model

In the next code cell, you will write a function that accepts a path to an image (such as `'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'`) as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model.  The output should always be an integer between 0 and 999, inclusive.

Before writing the function, make sure that you take the time to learn  how to appropriately pre-process tensors for pre-trained models in the [PyTorch documentation](http://pytorch.org/docs/stable/torchvision/models.html).


```python
from PIL import Image
import torchvision.transforms as transforms

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    image = Image.open(img_path)
    transformers = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.ToTensor()])
    image = transformers(image)
    
    image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    
    if use_cuda:
        image = image.cuda()
        
    output = VGG16(image)
    
    idx = torch.topk(output, 1)
    
    return idx[1] # predicted class index
```

### (IMPLEMENTATION) Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).

Use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    dog_idx = VGG16_predict(img_path)
    if dog_idx >= 151 and dog_idx <= 268:
        return True
    return False
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 2:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 

- 0% of dogs detected in human images
- 79% of dogs detected in dog images


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
is_dog_detected = 0
for image in human_files_short:
    if dog_detector(image):
        is_dog_detected += 1
        
print("Percetage of dogs detected in human images : ", is_dog_detected/len(human_files_short))
        
is_dog_detected = 0
for image in dog_files_short:
    if(dog_detector(image)):
        is_dog_detected += 1

print("Percentage of dogs detected in dog images : ", is_dog_detected/len(human_files_short))
```

    Percetage of dogs detected in human images :  0.0
    Percentage of dogs detected in dog images :  0.79


We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as [Inception-v3](http://pytorch.org/docs/master/torchvision/models.html#inception-v3), [ResNet-50](http://pytorch.org/docs/master/torchvision/models.html#id3), etc).  Please use the code cell below to test other pre-trained PyTorch models.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.


```python
### (Optional) 
### TODO: Report the performance of another pre-trained network.
### Feel free to use as many code cells as needed.
```

---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 10%.  In Step 4 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!

### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset

Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dog_images/train`, `dog_images/valid`, and `dog_images/test`, respectively).  You may find [this documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource.  If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!


```python
import os
from torchvision import datasets

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(128),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

other_transforms = transforms.Compose([transforms.Resize(150),
                                       transforms.CenterCrop(128),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder('/data/dog_images/train', transform=train_transforms)
test_data = datasets.ImageFolder('/data/dog_images/test', transform=other_transforms)
valid_data = datasets.ImageFolder('/data/dog_images/valid', transform=other_transforms)
```


```python
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=50)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50)

loaders_scratch = {'train' : train_loader,
                   'valid' : valid_loader,
                   'test' : test_loader}
```


```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

**Question 3:** Describe your chosen procedure for preprocessing the data. 
- How does your code resize the images (by cropping, stretching, etc)?  What size did you pick for the input tensor, and why?
- Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?


**Answer**:

- for preprocessing, I have used transform functions to resize the image, crop the image and convert it to tensor. Since images have three 255 color channel, I have use normalize transform to convert it to 0-1

- since images are of different size, I made them consistent of single size and computationally feasible I have resized them to 128

- Also, to make the learning/training invaiant to rotation and mirror images, I have used transforms for rotation and horizontal flip.


### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  Use the template in the code cell below.


```python
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        
        #first layer of CNN; input of size 128*128*3
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        #second layer of CNN; input of size 64*64*4
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        #third layer of CNN; input of size 32*32*8
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        #forth layer of CNN; input of size 16*16*16
        self.conv4 = nn.Conv2d(16, 64, 3, padding=1)
        
        #max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        #linear layer 64*8*8 -> 500
        self.fc1 = nn.Linear(64*8*8, 500)
        #linear layer 500 -> 133
        self.fc2 = nn.Linear(500, 133)
        
        #dropout p = 0.25
        self.dropout = nn.Dropout(0.25)
        
    
    def forward(self, x):
        ## Define forward behavior
        #add sequence of convolution and max pooling layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        #Flattent the input
        #x = x.view(-1, 64*8*8)
        x = x.view(-1, 64*8*8)
        #add dropout layer
        x = self.dropout(x)
        
        #add first hidden layer with relu activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        
        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch = model_scratch.cuda()
```

__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

__Answer:__ 

- I have created 4 conv nets to extract out the features
- one hidden and output layer
- first input layer is taking image of 128*128*3, using filter 3X3 filter and 4 layers with padding of 1 and stride is default 1, to prediect the feature.
- second layer takes input of 64*64*4, and again using 3X3 filter, 8 layers, padding of 1 and stride 1
- third layer takes input of 32*32*8
- forth layer takes input of 16*16*16
- output of conv layer is fed into relu activation function
- output of activation function is fed into max pool layer of size 2X2
- after conv layer setup, there are two linear layer, one hidden and one output layer
- hidden layer maps the output of conv layer to 500 hidden nodes, using relu activation function and dropout with 25% probability to regularize parameters
- output layers maps hidden layer output 500 to 133 classes of dog breed.

### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_scratch`, and the optimizer as `optimizer_scratch` below.


```python
import torch.optim as optim

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.1)
```

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_scratch.pt'`.


```python
import time
```


```python
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    #start_time = time.time()
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            
            output = model(data)
            
            loss = criterion(output, target)
            
            loss.backward()
            
            optimizer.step()
            train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            
            loss = criterion(output, target)
            
            valid_loss += (1 / (batch_idx + 1)) * (loss.data - valid_loss)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print("Saving mode state...")
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model
```


```python
# train the model
model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')
```

    Epoch: 1 	Training Loss: 4.886423 	Validation Loss: 4.879597 	Time Lapse: 97.509368
    Saving mode state...
    Epoch: 2 	Training Loss: 4.875259 	Validation Loss: 4.869941 	Time Lapse: 194.222693
    Saving mode state...
    Epoch: 3 	Training Loss: 4.864325 	Validation Loss: 4.852630 	Time Lapse: 290.983629
    Saving mode state...
    Epoch: 4 	Training Loss: 4.835687 	Validation Loss: 4.799258 	Time Lapse: 387.322449
    Saving mode state...
    Epoch: 5 	Training Loss: 4.795366 	Validation Loss: 4.722709 	Time Lapse: 484.433557
    Saving mode state...
    Epoch: 6 	Training Loss: 4.757916 	Validation Loss: 4.696942 	Time Lapse: 580.226829
    Saving mode state...
    Epoch: 7 	Training Loss: 4.741291 	Validation Loss: 4.693686 	Time Lapse: 676.327821
    Saving mode state...
    Epoch: 8 	Training Loss: 4.709776 	Validation Loss: 4.674484 	Time Lapse: 772.321360
    Saving mode state...
    Epoch: 9 	Training Loss: 4.694751 	Validation Loss: 4.590972 	Time Lapse: 867.847887
    Saving mode state...
    Epoch: 10 	Training Loss: 4.650723 	Validation Loss: 4.612731 	Time Lapse: 963.975567
    Epoch: 11 	Training Loss: 4.599170 	Validation Loss: 4.502723 	Time Lapse: 1060.752949
    Saving mode state...
    Epoch: 12 	Training Loss: 4.553179 	Validation Loss: 4.391983 	Time Lapse: 1156.809605
    Saving mode state...
    Epoch: 13 	Training Loss: 4.515937 	Validation Loss: 4.374467 	Time Lapse: 1253.126225
    Saving mode state...
    Epoch: 14 	Training Loss: 4.482012 	Validation Loss: 4.375562 	Time Lapse: 1350.042818
    Epoch: 15 	Training Loss: 4.463867 	Validation Loss: 4.245409 	Time Lapse: 1446.546535
    Saving mode state...
    Epoch: 16 	Training Loss: 4.454102 	Validation Loss: 4.338253 	Time Lapse: 1542.613452
    Epoch: 17 	Training Loss: 4.418071 	Validation Loss: 4.240673 	Time Lapse: 1639.331269
    Saving mode state...
    Epoch: 18 	Training Loss: 4.396289 	Validation Loss: 4.387280 	Time Lapse: 1735.331970
    Epoch: 19 	Training Loss: 4.360408 	Validation Loss: 4.211399 	Time Lapse: 1831.048695
    Saving mode state...
    Epoch: 20 	Training Loss: 4.338610 	Validation Loss: 4.148777 	Time Lapse: 1926.923929
    Saving mode state...
    Epoch: 21 	Training Loss: 4.327539 	Validation Loss: 4.232719 	Time Lapse: 2023.523237
    Epoch: 22 	Training Loss: 4.304200 	Validation Loss: 4.123589 	Time Lapse: 2119.712989
    Saving mode state...
    Epoch: 23 	Training Loss: 4.279877 	Validation Loss: 4.127640 	Time Lapse: 2215.681163
    Epoch: 24 	Training Loss: 4.266626 	Validation Loss: 4.074350 	Time Lapse: 2311.799604
    Saving mode state...
    Epoch: 25 	Training Loss: 4.245734 	Validation Loss: 4.068345 	Time Lapse: 2408.574106
    Saving mode state...
    Epoch: 26 	Training Loss: 4.229381 	Validation Loss: 4.082512 	Time Lapse: 2504.238770
    Epoch: 27 	Training Loss: 4.215263 	Validation Loss: 4.182200 	Time Lapse: 2600.587992
    Epoch: 28 	Training Loss: 4.180215 	Validation Loss: 4.059848 	Time Lapse: 2696.469628
    Saving mode state...
    Epoch: 29 	Training Loss: 4.144912 	Validation Loss: 4.110093 	Time Lapse: 2792.271753
    Epoch: 30 	Training Loss: 4.155279 	Validation Loss: 4.034084 	Time Lapse: 2887.715825
    Saving mode state...
    Epoch: 31 	Training Loss: 4.112092 	Validation Loss: 4.124443 	Time Lapse: 2983.363064
    Epoch: 32 	Training Loss: 4.100640 	Validation Loss: 3.904062 	Time Lapse: 3078.888630
    Saving mode state...
    Epoch: 33 	Training Loss: 4.099453 	Validation Loss: 3.952647 	Time Lapse: 3173.854712
    Epoch: 34 	Training Loss: 4.044452 	Validation Loss: 3.895574 	Time Lapse: 3268.698834
    Saving mode state...
    Epoch: 35 	Training Loss: 4.050308 	Validation Loss: 3.962309 	Time Lapse: 3363.558568
    Epoch: 36 	Training Loss: 4.017183 	Validation Loss: 3.927063 	Time Lapse: 3460.402596
    Epoch: 37 	Training Loss: 4.006762 	Validation Loss: 3.888601 	Time Lapse: 3557.585404
    Saving mode state...
    Epoch: 38 	Training Loss: 3.987502 	Validation Loss: 3.872988 	Time Lapse: 3654.428132
    Saving mode state...
    Epoch: 39 	Training Loss: 3.970101 	Validation Loss: 3.829613 	Time Lapse: 3750.655365
    Saving mode state...
    Epoch: 40 	Training Loss: 3.944451 	Validation Loss: 3.848987 	Time Lapse: 3847.789136
    Epoch: 41 	Training Loss: 3.927730 	Validation Loss: 4.178623 	Time Lapse: 3943.594022
    Epoch: 42 	Training Loss: 3.924643 	Validation Loss: 3.796447 	Time Lapse: 4040.585537
    Saving mode state...
    Epoch: 43 	Training Loss: 3.901126 	Validation Loss: 3.850577 	Time Lapse: 4137.651520
    Epoch: 44 	Training Loss: 3.868905 	Validation Loss: 3.749395 	Time Lapse: 4234.321883
    Saving mode state...
    Epoch: 45 	Training Loss: 3.863081 	Validation Loss: 3.805774 	Time Lapse: 4331.566214
    Epoch: 46 	Training Loss: 3.824536 	Validation Loss: 3.792345 	Time Lapse: 4428.041528
    Epoch: 47 	Training Loss: 3.810031 	Validation Loss: 3.774614 	Time Lapse: 4524.135439
    Epoch: 48 	Training Loss: 3.817621 	Validation Loss: 3.805110 	Time Lapse: 4619.084632
    Epoch: 49 	Training Loss: 3.825442 	Validation Loss: 3.744512 	Time Lapse: 4714.506952
    Saving mode state...
    Epoch: 50 	Training Loss: 3.793214 	Validation Loss: 3.742616 	Time Lapse: 4810.259377
    Saving mode state...
    Epoch: 51 	Training Loss: 3.766177 	Validation Loss: 3.798465 	Time Lapse: 4905.914566
    Epoch: 52 	Training Loss: 3.761077 	Validation Loss: 3.835851 	Time Lapse: 5001.517230
    Epoch: 53 	Training Loss: 3.758261 	Validation Loss: 3.598266 	Time Lapse: 5096.760610
    Saving mode state...
    Epoch: 54 	Training Loss: 3.747186 	Validation Loss: 3.979785 	Time Lapse: 5191.711038
    Epoch: 55 	Training Loss: 3.741364 	Validation Loss: 3.604705 	Time Lapse: 5286.621716
    Epoch: 56 	Training Loss: 3.704821 	Validation Loss: 3.722239 	Time Lapse: 5381.260272
    Epoch: 57 	Training Loss: 3.698728 	Validation Loss: 3.686069 	Time Lapse: 5477.312291
    Epoch: 58 	Training Loss: 3.694177 	Validation Loss: 3.593388 	Time Lapse: 5572.912134
    Saving mode state...
    Epoch: 59 	Training Loss: 3.690163 	Validation Loss: 3.799825 	Time Lapse: 5669.486465
    Epoch: 60 	Training Loss: 3.663173 	Validation Loss: 3.573985 	Time Lapse: 5765.809086
    Saving mode state...
    Epoch: 61 	Training Loss: 3.648988 	Validation Loss: 3.641426 	Time Lapse: 5862.742615
    Epoch: 62 	Training Loss: 3.636669 	Validation Loss: 3.832028 	Time Lapse: 5959.779861
    Epoch: 63 	Training Loss: 3.640197 	Validation Loss: 3.616022 	Time Lapse: 6057.253762
    Epoch: 64 	Training Loss: 3.640662 	Validation Loss: 3.536354 	Time Lapse: 6152.973564
    Saving mode state...
    Epoch: 65 	Training Loss: 3.609437 	Validation Loss: 3.597948 	Time Lapse: 6248.590455
    Epoch: 66 	Training Loss: 3.581224 	Validation Loss: 3.811439 	Time Lapse: 6343.953939
    Epoch: 67 	Training Loss: 3.606814 	Validation Loss: 3.602118 	Time Lapse: 6439.416136
    Epoch: 68 	Training Loss: 3.591333 	Validation Loss: 3.633021 	Time Lapse: 6534.902418
    Epoch: 69 	Training Loss: 3.571041 	Validation Loss: 3.602787 	Time Lapse: 6630.276215
    Epoch: 70 	Training Loss: 3.567028 	Validation Loss: 3.582231 	Time Lapse: 6726.243735
    Epoch: 71 	Training Loss: 3.539308 	Validation Loss: 3.547816 	Time Lapse: 6821.170757
    Epoch: 72 	Training Loss: 3.556515 	Validation Loss: 3.593282 	Time Lapse: 6917.550472
    Epoch: 73 	Training Loss: 3.542970 	Validation Loss: 3.532743 	Time Lapse: 7013.202750
    Saving mode state...
    Epoch: 74 	Training Loss: 3.521019 	Validation Loss: 3.811149 	Time Lapse: 7108.525530
    Epoch: 75 	Training Loss: 3.547502 	Validation Loss: 3.752453 	Time Lapse: 7203.273412
    Epoch: 76 	Training Loss: 3.511844 	Validation Loss: 3.565132 	Time Lapse: 7298.221903
    Epoch: 77 	Training Loss: 3.517002 	Validation Loss: 3.621006 	Time Lapse: 7393.707220
    Epoch: 78 	Training Loss: 3.520350 	Validation Loss: 3.532988 	Time Lapse: 7488.509989
    Epoch: 79 	Training Loss: 3.497441 	Validation Loss: 3.491024 	Time Lapse: 7583.700331
    Saving mode state...
    Epoch: 80 	Training Loss: 3.501848 	Validation Loss: 3.681601 	Time Lapse: 7679.429549
    Epoch: 81 	Training Loss: 3.488699 	Validation Loss: 3.519215 	Time Lapse: 7774.345393
    Epoch: 82 	Training Loss: 3.475754 	Validation Loss: 3.601668 	Time Lapse: 7868.955407
    Epoch: 83 	Training Loss: 3.460318 	Validation Loss: 3.754809 	Time Lapse: 7964.249469
    Epoch: 84 	Training Loss: 3.478374 	Validation Loss: 3.450270 	Time Lapse: 8060.119858
    Saving mode state...
    Epoch: 85 	Training Loss: 3.431526 	Validation Loss: 3.463316 	Time Lapse: 8155.401801
    Epoch: 86 	Training Loss: 3.439894 	Validation Loss: 3.583519 	Time Lapse: 8251.562285
    Epoch: 87 	Training Loss: 3.439465 	Validation Loss: 3.737397 	Time Lapse: 8347.829859
    Epoch: 88 	Training Loss: 3.422736 	Validation Loss: 3.492920 	Time Lapse: 8444.200393
    Epoch: 89 	Training Loss: 3.416250 	Validation Loss: 3.557446 	Time Lapse: 8539.542477
    Epoch: 90 	Training Loss: 3.400147 	Validation Loss: 3.627806 	Time Lapse: 8634.540845
    Epoch: 91 	Training Loss: 3.445681 	Validation Loss: 3.568504 	Time Lapse: 8729.286474
    Epoch: 92 	Training Loss: 3.414209 	Validation Loss: 3.750311 	Time Lapse: 8824.219975
    Epoch: 93 	Training Loss: 3.451576 	Validation Loss: 3.511608 	Time Lapse: 8919.679157
    Epoch: 94 	Training Loss: 3.388504 	Validation Loss: 3.627197 	Time Lapse: 9014.862975
    Epoch: 95 	Training Loss: 3.418049 	Validation Loss: 3.439137 	Time Lapse: 9110.736634
    Saving mode state...
    Epoch: 96 	Training Loss: 3.401429 	Validation Loss: 3.566561 	Time Lapse: 9205.980105
    Epoch: 97 	Training Loss: 3.378850 	Validation Loss: 3.527889 	Time Lapse: 9301.196228
    Epoch: 98 	Training Loss: 3.367525 	Validation Loss: 3.424206 	Time Lapse: 9396.634902
    Saving mode state...
    Epoch: 99 	Training Loss: 3.377968 	Validation Loss: 3.580671 	Time Lapse: 9491.649729
    Epoch: 100 	Training Loss: 3.331451 	Validation Loss: 3.513570 	Time Lapse: 9586.698925



```python
# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images.  Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 10%.


```python
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
```


```python
# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)
```

    Test Loss: 3.432597
    
    
    Test Accuracy: 18% (157/836)


---
<a id='step4'></a>
## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset

Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively). 

If you like, **you are welcome to use the same data loaders from the previous step**, when you created a CNN from scratch.


```python
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

other_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder('/data/dog_images/train', transform=train_transforms)
test_data = datasets.ImageFolder('/data/dog_images/test', transform=other_transforms)
valid_data = datasets.ImageFolder('/data/dog_images/valid', transform=other_transforms)

## TODO: Specify data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=50)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50)

loaders_transfer = {'train' : train_loader,
                   'valid' : valid_loader,
                   'test' : test_loader}
```

### (IMPLEMENTATION) Model Architecture

Use transfer learning to create a CNN to classify dog breed.  Use the code cell below, and save your initialized model as the variable `model_transfer`.


```python
## TODO: Specify model architecture 
model_transfer = models.vgg16(pretrained=True)
for param in model_transfer.features.parameters():
    param.requires_grad_(False)
    
n_inputs = model_transfer.classifier[6].in_features

last_layer = nn.Linear(n_inputs, 133)

model_transfer.classifier[6] = last_layer

if use_cuda:
    model_transfer = model_transfer.cuda()
```

__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ 

- I am using vgg16 pretrained network
- VGG16 network has 1000 output classed to map it to our dog breed classifier I have updated the last output layer of classifier with 133 nodes linear layer.
- 

### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html).  Save the chosen loss function as `criterion_transfer`, and the optimizer as `optimizer_transfer` below.


```python
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.1)
```

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.


```python
# train the model
model_transfer = train(100, loaders_transfer, model_transfer, optimizer_transfer, 
                       criterion_transfer, use_cuda, 'model_transfer.pt')

```

    Epoch: 1 	Training Loss: 2.455689 	Validation Loss: 1.274142
    Saving mode state...
    Epoch: 2 	Training Loss: 1.940873 	Validation Loss: 1.294513
    Epoch: 3 	Training Loss: 1.920743 	Validation Loss: 1.172062
    Saving mode state...
    Epoch: 4 	Training Loss: 1.948402 	Validation Loss: 1.450789
    Epoch: 5 	Training Loss: 1.926047 	Validation Loss: 1.163601
    Saving mode state...
    Epoch: 6 	Training Loss: 1.977927 	Validation Loss: 1.233286
    Epoch: 7 	Training Loss: 1.980088 	Validation Loss: 1.210694
    Epoch: 8 	Training Loss: 1.934350 	Validation Loss: 1.529511
    Epoch: 9 	Training Loss: 2.016789 	Validation Loss: 1.453657
    Epoch: 10 	Training Loss: 2.026618 	Validation Loss: 1.091981
    Saving mode state...
    Epoch: 11 	Training Loss: 2.064902 	Validation Loss: 1.429709
    Epoch: 12 	Training Loss: 2.115121 	Validation Loss: 1.327634
    Epoch: 13 	Training Loss: 2.100820 	Validation Loss: 1.157226
    Epoch: 14 	Training Loss: 2.203647 	Validation Loss: 1.214244
    Epoch: 15 	Training Loss: 2.168800 	Validation Loss: 1.501287
    Epoch: 16 	Training Loss: 2.238090 	Validation Loss: 1.317222
    Epoch: 17 	Training Loss: 2.311007 	Validation Loss: 1.699865
    Epoch: 18 	Training Loss: 2.305017 	Validation Loss: 1.163026
    Epoch: 19 	Training Loss: 2.426359 	Validation Loss: 1.561262
    Epoch: 20 	Training Loss: 2.584028 	Validation Loss: 1.852819
    Epoch: 21 	Training Loss: 2.599945 	Validation Loss: 2.584747
    Epoch: 22 	Training Loss: 2.836982 	Validation Loss: 1.684002
    Epoch: 23 	Training Loss: 2.763401 	Validation Loss: 1.565100
    Epoch: 24 	Training Loss: 2.822858 	Validation Loss: 1.811316
    Epoch: 25 	Training Loss: 2.911371 	Validation Loss: 1.737548
    Epoch: 26 	Training Loss: 2.919921 	Validation Loss: 1.636680
    Epoch: 27 	Training Loss: 3.127911 	Validation Loss: 1.894851
    Epoch: 28 	Training Loss: 3.263442 	Validation Loss: 2.814782
    Epoch: 29 	Training Loss: 3.483766 	Validation Loss: 2.192627
    Epoch: 30 	Training Loss: 3.563375 	Validation Loss: 2.533426
    Epoch: 31 	Training Loss: 3.689139 	Validation Loss: 3.071112
    Epoch: 32 	Training Loss: 3.934777 	Validation Loss: 3.447416
    Epoch: 33 	Training Loss: 4.344285 	Validation Loss: 4.027645
    Epoch: 34 	Training Loss: 4.671786 	Validation Loss: 3.002820
    Epoch: 35 	Training Loss: 4.586733 	Validation Loss: 4.393607
    Epoch: 36 	Training Loss: 4.815905 	Validation Loss: 5.612746
    Epoch: 37 	Training Loss: 5.076232 	Validation Loss: 3.596058
    Epoch: 38 	Training Loss: 5.184298 	Validation Loss: 3.384761
    Epoch: 39 	Training Loss: 5.354546 	Validation Loss: 4.125442
    Epoch: 40 	Training Loss: 5.109061 	Validation Loss: 2.990754
    Epoch: 41 	Training Loss: 5.240492 	Validation Loss: 5.272402
    Epoch: 42 	Training Loss: 5.658452 	Validation Loss: 3.261463
    Epoch: 43 	Training Loss: 5.544466 	Validation Loss: 3.481131
    Epoch: 44 	Training Loss: 5.101726 	Validation Loss: 3.918122
    Epoch: 45 	Training Loss: 5.387646 	Validation Loss: 4.025811
    Epoch: 46 	Training Loss: 5.219089 	Validation Loss: 3.576148
    Epoch: 47 	Training Loss: 5.058558 	Validation Loss: 3.195469
    Epoch: 48 	Training Loss: 5.083657 	Validation Loss: 3.513838
    Epoch: 49 	Training Loss: 5.047402 	Validation Loss: 3.854337
    Epoch: 50 	Training Loss: 4.836813 	Validation Loss: 3.373040
    Epoch: 51 	Training Loss: 4.554022 	Validation Loss: 3.710722
    Epoch: 52 	Training Loss: 4.598453 	Validation Loss: 3.448506



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-17-8138f0e26e41> in <module>()
          1 # train the model
          2 model_transfer = train(100, loaders_transfer, model_transfer, optimizer_transfer, 
    ----> 3                        criterion_transfer, use_cuda, 'model_transfer.pt')
    

    <ipython-input-12-221cf9a28bfd> in train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path)
         14         ###################
         15         model.train()
    ---> 16         for batch_idx, (data, target) in enumerate(loaders['train']):
         17             # move to GPU
         18             if use_cuda:


    /opt/conda/lib/python3.6/site-packages/torch/utils/data/dataloader.py in __next__(self)
        262         if self.num_workers == 0:  # same-process loading
        263             indices = next(self.sample_iter)  # may raise StopIteration
    --> 264             batch = self.collate_fn([self.dataset[i] for i in indices])
        265             if self.pin_memory:
        266                 batch = pin_memory_batch(batch)


    /opt/conda/lib/python3.6/site-packages/torch/utils/data/dataloader.py in <listcomp>(.0)
        262         if self.num_workers == 0:  # same-process loading
        263             indices = next(self.sample_iter)  # may raise StopIteration
    --> 264             batch = self.collate_fn([self.dataset[i] for i in indices])
        265             if self.pin_memory:
        266                 batch = pin_memory_batch(batch)


    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/datasets/folder.py in __getitem__(self, index)
        101         sample = self.loader(path)
        102         if self.transform is not None:
    --> 103             sample = self.transform(sample)
        104         if self.target_transform is not None:
        105             target = self.target_transform(target)


    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/transforms/transforms.py in __call__(self, img)
         47     def __call__(self, img):
         48         for t in self.transforms:
    ---> 49             img = t(img)
         50         return img
         51 


    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/transforms/transforms.py in __call__(self, img)
        544         """
        545         i, j, h, w = self.get_params(img, self.scale, self.ratio)
    --> 546         return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        547 
        548     def __repr__(self):


    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/transforms/functional.py in resized_crop(img, i, j, h, w, size, interpolation)
        329     assert _is_pil_image(img), 'img should be PIL Image'
        330     img = crop(img, i, j, h, w)
    --> 331     img = resize(img, size, interpolation)
        332     return img
        333 


    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/transforms/functional.py in resize(img, size, interpolation)
        204             return img.resize((ow, oh), interpolation)
        205     else:
    --> 206         return img.resize(size[::-1], interpolation)
        207 
        208 


    /opt/conda/lib/python3.6/site-packages/PIL/Image.py in resize(self, size, resample)
       1710             return self.convert('RGBa').resize(size, resample).convert('RGBA')
       1711 
    -> 1712         return self._new(self.im.resize(size, resample))
       1713 
       1714     def rotate(self, angle, resample=NEAREST, expand=0, center=None,


    KeyboardInterrupt: 



```python
#as validation loss is increasing, decrease learning rate and try again
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.01)
```


```python
# train the model
model_transfer = train(100, loaders_transfer, model_transfer, optimizer_transfer, 
                       criterion_transfer, use_cuda, 'model_transfer.pt')
```

    Epoch: 1 	Training Loss: 3.152009 	Validation Loss: 1.058426
    Saving mode state...
    Epoch: 2 	Training Loss: 1.696960 	Validation Loss: 0.728885
    Saving mode state...
    Epoch: 3 	Training Loss: 1.446295 	Validation Loss: 0.627003
    Saving mode state...
    Epoch: 4 	Training Loss: 1.307494 	Validation Loss: 0.560708
    Saving mode state...
    Epoch: 5 	Training Loss: 1.251254 	Validation Loss: 0.540710
    Saving mode state...
    Epoch: 6 	Training Loss: 1.183810 	Validation Loss: 0.560675
    Epoch: 7 	Training Loss: 1.165184 	Validation Loss: 0.531526
    Saving mode state...
    Epoch: 8 	Training Loss: 1.114684 	Validation Loss: 0.503925
    Saving mode state...
    Epoch: 9 	Training Loss: 1.098078 	Validation Loss: 0.494864
    Saving mode state...
    Epoch: 10 	Training Loss: 1.063930 	Validation Loss: 0.511608
    Epoch: 11 	Training Loss: 1.048267 	Validation Loss: 0.470766
    Saving mode state...
    Epoch: 12 	Training Loss: 1.019256 	Validation Loss: 0.447114
    Saving mode state...
    Epoch: 13 	Training Loss: 0.995369 	Validation Loss: 0.478794
    Epoch: 14 	Training Loss: 0.988689 	Validation Loss: 0.454299
    Epoch: 15 	Training Loss: 0.963335 	Validation Loss: 0.476475
    Epoch: 16 	Training Loss: 0.944941 	Validation Loss: 0.464611
    Epoch: 17 	Training Loss: 0.965491 	Validation Loss: 0.471863
    Epoch: 18 	Training Loss: 0.940168 	Validation Loss: 0.462715
    Epoch: 19 	Training Loss: 0.914118 	Validation Loss: 0.454721
    Epoch: 20 	Training Loss: 0.910591 	Validation Loss: 0.466814
    Epoch: 21 	Training Loss: 0.898766 	Validation Loss: 0.440272
    Saving mode state...
    Epoch: 22 	Training Loss: 0.874271 	Validation Loss: 0.433228
    Saving mode state...
    Epoch: 23 	Training Loss: 0.866224 	Validation Loss: 0.472960
    Epoch: 24 	Training Loss: 0.867791 	Validation Loss: 0.466433
    Epoch: 25 	Training Loss: 0.861278 	Validation Loss: 0.477725
    Epoch: 26 	Training Loss: 0.857228 	Validation Loss: 0.446536
    Epoch: 27 	Training Loss: 0.843961 	Validation Loss: 0.449855
    Epoch: 28 	Training Loss: 0.829871 	Validation Loss: 0.460411
    Epoch: 29 	Training Loss: 0.817799 	Validation Loss: 0.452895
    Epoch: 30 	Training Loss: 0.829997 	Validation Loss: 0.457898
    Epoch: 31 	Training Loss: 0.825993 	Validation Loss: 0.449323
    Epoch: 32 	Training Loss: 0.788293 	Validation Loss: 0.472220
    Epoch: 33 	Training Loss: 0.792890 	Validation Loss: 0.444216
    Epoch: 34 	Training Loss: 0.803514 	Validation Loss: 0.450424
    Epoch: 35 	Training Loss: 0.814693 	Validation Loss: 0.447221
    Epoch: 36 	Training Loss: 0.769778 	Validation Loss: 0.426506
    Saving mode state...
    Epoch: 37 	Training Loss: 0.790674 	Validation Loss: 0.424186
    Saving mode state...
    Epoch: 38 	Training Loss: 0.799400 	Validation Loss: 0.441196
    Epoch: 39 	Training Loss: 0.790648 	Validation Loss: 0.431818
    Epoch: 40 	Training Loss: 0.758480 	Validation Loss: 0.463019
    Epoch: 41 	Training Loss: 0.766968 	Validation Loss: 0.457511
    Epoch: 42 	Training Loss: 0.743545 	Validation Loss: 0.450446
    Epoch: 43 	Training Loss: 0.745661 	Validation Loss: 0.444686
    Epoch: 44 	Training Loss: 0.769765 	Validation Loss: 0.455225
    Epoch: 45 	Training Loss: 0.739360 	Validation Loss: 0.444540
    Epoch: 46 	Training Loss: 0.759189 	Validation Loss: 0.458873
    Epoch: 47 	Training Loss: 0.745268 	Validation Loss: 0.462381
    Epoch: 48 	Training Loss: 0.743818 	Validation Loss: 0.458137
    Epoch: 49 	Training Loss: 0.754545 	Validation Loss: 0.437504
    Epoch: 50 	Training Loss: 0.713507 	Validation Loss: 0.439214
    Epoch: 51 	Training Loss: 0.728684 	Validation Loss: 0.436253
    Epoch: 52 	Training Loss: 0.715039 	Validation Loss: 0.453814
    Epoch: 53 	Training Loss: 0.703409 	Validation Loss: 0.440115
    Epoch: 54 	Training Loss: 0.701425 	Validation Loss: 0.440532
    Epoch: 55 	Training Loss: 0.705839 	Validation Loss: 0.445215
    Epoch: 56 	Training Loss: 0.678119 	Validation Loss: 0.463915
    Epoch: 57 	Training Loss: 0.687678 	Validation Loss: 0.435551
    Epoch: 58 	Training Loss: 0.686591 	Validation Loss: 0.448203
    Epoch: 59 	Training Loss: 0.699062 	Validation Loss: 0.443605
    Epoch: 60 	Training Loss: 0.698570 	Validation Loss: 0.435776
    Epoch: 61 	Training Loss: 0.688472 	Validation Loss: 0.440731
    Epoch: 62 	Training Loss: 0.704427 	Validation Loss: 0.439761
    Epoch: 63 	Training Loss: 0.700710 	Validation Loss: 0.445615
    Epoch: 64 	Training Loss: 0.665817 	Validation Loss: 0.458470
    Epoch: 65 	Training Loss: 0.674946 	Validation Loss: 0.436300


### workspace got disconnected after 65 epochs. as we can observe we have reached some good minimum validation loss on 42nd epoch and there is not min validation loss till 65th epoch so lets see the test accuray on the model saved at 42nd epoch


```python
# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt', map_location=lambda storage, loc: storage))
if use_cuda:
    model_transfer = model_transfer.cuda()
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.


```python
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
```

    Test Loss: 0.447515
    
    
    Test Accuracy: 86% (721/836)


### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan hound`, etc) that is predicted by your model.  


```python
#class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].classes]
classes = {}

for _class in np.array(glob("/data/dog_images/train/*/")):
    idx, name = _class.split('/')[4].split('.')
    classes[int(idx)] = name
print(classes)
```

    {103: 'Mastiff', 59: 'Doberman_pinscher', 55: 'Curly-coated_retriever', 31: 'Borzoi', 24: 'Bichon_frise', 49: 'Chinese_crested', 67: 'Finnish_spitz', 130: 'Welsh_springer_spaniel', 19: 'Bedlington_terrier', 115: 'Papillon', 126: 'Saint_bernard', 13: 'Australian_terrier', 116: 'Parson_russell_terrier', 107: 'Norfolk_terrier', 133: 'Yorkshire_terrier', 32: 'Boston_terrier', 108: 'Norwegian_buhund', 28: 'Bluetick_coonhound', 66: 'Field_spaniel', 129: 'Tibetan_mastiff', 5: 'Alaskan_malamute', 102: 'Manchester_terrier', 34: 'Boxer', 68: 'Flat-coated_retriever', 89: 'Irish_wolfhound', 104: 'Miniature_schnauzer', 35: 'Boykin_spaniel', 80: 'Greater_swiss_mountain_dog', 7: 'American_foxhound', 112: 'Nova_scotia_duck_tolling_retriever', 25: 'Black_and_tan_coonhound', 72: 'German_shorthaired_pointer', 62: 'English_setter', 29: 'Border_collie', 45: 'Cardigan_welsh_corgi', 105: 'Neapolitan_mastiff', 76: 'Golden_retriever', 63: 'English_springer_spaniel', 78: 'Great_dane', 84: 'Icelandic_sheepdog', 23: 'Bernese_mountain_dog', 91: 'Japanese_chin', 11: 'Australian_cattle_dog', 21: 'Belgian_sheepdog', 41: 'Bullmastiff', 98: 'Leonberger', 18: 'Beauceron', 20: 'Belgian_malinois', 16: 'Beagle', 39: 'Bull_terrier', 87: 'Irish_terrier', 64: 'English_toy_spaniel', 123: 'Pomeranian', 97: 'Lakeland_terrier', 127: 'Silky_terrier', 120: 'Pharaoh_hound', 6: 'American_eskimo_dog', 12: 'Australian_shepherd', 70: 'German_pinscher', 95: 'Kuvasz', 131: 'Wirehaired_pointing_griffon', 125: 'Portuguese_water_dog', 71: 'German_shepherd_dog', 3: 'Airedale_terrier', 43: 'Canaan_dog', 118: 'Pembroke_welsh_corgi', 10: 'Anatolian_shepherd_dog', 33: 'Bouvier_des_flandres', 106: 'Newfoundland', 47: 'Chesapeake_bay_retriever', 9: 'American_water_spaniel', 65: 'Entlebucher_mountain_dog', 2: 'Afghan_hound', 54: 'Collie', 93: 'Kerry_blue_terrier', 61: 'English_cocker_spaniel', 82: 'Havanese', 44: 'Cane_corso', 56: 'Dachshund', 26: 'Black_russian_terrier', 132: 'Xoloitzcuintli', 94: 'Komondor', 22: 'Belgian_tervuren', 114: 'Otterhound', 36: 'Briard', 74: 'Giant_schnauzer', 17: 'Bearded_collie', 110: 'Norwegian_lundehund', 85: 'Irish_red_and_white_setter', 69: 'French_bulldog', 75: 'Glen_of_imaal_terrier', 42: 'Cairn_terrier', 4: 'Akita', 60: 'Dogue_de_bordeaux', 128: 'Smooth_fox_terrier', 83: 'Ibizan_hound', 117: 'Pekingese', 81: 'Greyhound', 51: 'Chow_chow', 40: 'Bulldog', 8: 'American_staffordshire_terrier', 46: 'Cavalier_king_charles_spaniel', 99: 'Lhasa_apso', 90: 'Italian_greyhound', 50: 'Chinese_shar-pei', 86: 'Irish_setter', 37: 'Brittany', 121: 'Plott', 14: 'Basenji', 30: 'Border_terrier', 79: 'Great_pyrenees', 96: 'Labrador_retriever', 27: 'Bloodhound', 48: 'Chihuahua', 119: 'Petit_basset_griffon_vendeen', 124: 'Poodle', 58: 'Dandie_dinmont_terrier', 52: 'Clumber_spaniel', 38: 'Brussels_griffon', 113: 'Old_english_sheepdog', 57: 'Dalmatian', 53: 'Cocker_spaniel', 122: 'Pointer', 77: 'Gordon_setter', 73: 'German_wirehaired_pointer', 88: 'Irish_water_spaniel', 111: 'Norwich_terrier', 109: 'Norwegian_elkhound', 1: 'Affenpinscher', 15: 'Basset_hound', 101: 'Maltese', 92: 'Keeshond', 100: 'Lowchen'}



```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
#class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].classes]



def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    image = Image.open(img_path)
    transformers = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.ToTensor()])
    image = transformers(image)
    
    image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    
    if use_cuda:
        image = image.cuda()
        
    output = model_transfer(image)
    
    idx = torch.topk(output, 1)
    
    return classes[idx[1].item()]
```


```python
human_files_short = human_files[:100]
dog_files_short = dog_files[:100]
predict_breed_transfer(dog_files_short[20])
```




    'Manchester_terrier'



---
<a id='step5'></a>
## Step 5: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `human_detector` functions developed above.  You are __required__ to use your CNN from Step 4 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    img = mpimg.imread(img_path)
        
    #use face detector to see if image contains face
    is_human = face_detector(img_path)
    if face_detector(img_path):
        #human is detected
        plt.imshow(img)
        plt.show()
        breed = predict_breed_transfer(img_path)
        print('human is detected')
        print('human looks like : ', breed)
    elif dog_detector(img_path):
        #dog detected
        plt.imshow(img)
        plt.show()
        breed = predict_breed_transfer(img_path)
        print('dog is detected')
        print('dog looks like : ', breed)
    else:
        plt.imshow(img)
        plt.show()
        print('no human or dog is detected')
```

---
<a id='step6'></a>
## Step 6: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that _you_ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ 

- More training with smaller learning rate as with current learning rate, it is observed that validation error is kind of stable or increasing after 50 epochs
- User other model than vgg16 to achieve high accuracy
- Deep learning model can be used to detect human face


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

## suggested code, below
for file in np.hstack((human_files[:3], dog_files[:3])):
    run_app(file)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-112491319529> in <module>()
          4 
          5 ## suggested code, below
    ----> 6 for file in np.hstack((human_files[:3], dog_files[:3])):
          7     run_app(file)


    NameError: name 'np' is not defined



```python

```
