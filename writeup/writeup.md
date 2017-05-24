# Project 5 - Vehicle Detection


---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/carnotcar.png
[image2]: ./images/hogCar.png
[image3]: ./images/hogNotCar.png
[image4]: ./images/roi64.png
[image5]: ./images/roi96.png
[image6]: ./images/roi128.png
[image7]: ./images/roi160.png
[image8]: ./images/carWindows.png
[image9]: ./images/heatmap.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###  Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 'HOG Transform' section under 'Feature Extraction' in the [Classifier Workbook](https://github.com/scrottty/CarND-Vehicle-Detection/blob/master/workbook_Classifier.ipynb)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

##### HOG FEATURES

Here is an example using the `LAB` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

Initially I played around with various combnations of parameters to produce a Hog image that i thought pulled the best features form the model. However, realising that the final classifier accuracy was the most important output from this, I tuned based  upon that instead of what looked 'best' for me

The final choosen HOG features paramters where:

```py
orient = 12
pixels_per_cell = (8,8)
cells_per_block = (2,2)
hog_channel = 'ALL'
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for the classifier can be found in the 'Build Classifier' section in the [Classifier Workbook](https://github.com/scrottty/CarND-Vehicle-Detection/blob/master/workbook_Classifier.ipynb)

I trained a linear SVM using the `sklearn` library to produce a classifier with a final accuracy of 99.33%.

The process to produce this was:
1. Load the datasets to be used
2. Extract the features for the Linear SVM to be tuned on. The features were:
    * Spatial features of a 32x32 resolution image in the `LAB` colorspace
    * Histograms features of the image in the `LAB` colorspace
    * HOG features of all channels in the `LAB` colorspace with the paramters as mentioned above
3. Scale all the features to normalise them to one another. This allows for the classifier to use the features evenly
4. Generate labels for the images
5. Randomly shuffle the data into training and testing sets with a 80/20 split
6. Fit the Linear SVM to the training set
7. Asses the accuracy of the model on the testing set
9. Tune Hyperparamters to produce the best base level classifier
8. Adjust parameters to find best combination to produce the best classifier accuracy

This was done iteratively with adjustments made one by one to find the best classifier. A record of the changes can be seen in [this Excel File](https://github.com/scrottty/CarND-Vehicle-Detection/blob/master/SDD.xlsx)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The slinding window implimentation can be found in [lessonFunctions.py](https://github.com/scrottty/CarND-Vehicle-Detection/blob/master/lessonFunctions.py)

I implemented a sliding window approach to pull 64x64 frames to pass to the classifier to be classified. To account for the changing perspective of the cars in the videos (as they get bigger/ smaller as they get closer/further away) different windows sizes were used then scaled to produce a 64x64 window

The sliding window was done in the following steps:
1. Regions of interest and its respective window size were selected
2. Each region of interested is pulled from the image and scalled to fit a 64x64 window. This was done to match the parameters of the HOG features
3. For each region of interest and window size the windows to search were created through the slding window approach. This calculated the pixel positions in the image to produce windows of a 64x64 to search. An `overlap = 0.8` was chosen. This was done to produce as many successul window classifications for the cars as possible so to make it easier to filter out the noise as explained further on. This had the trade off of slowing the pipeline down as there were now more windows to process however produce a much better result

Later on I realised that I had accidentally selected regions of interest that ignored the left lane. For this video where the car is driving in the left lane it works fine. However to better generalise it would be best to remove this bias and have the regions of interest across the image.

Exmaples of the Regions of Interest and the windows are shown. Note the overlap is set to 0 for demonstration purposes

![alt text][image4]
![alt text][image5]

![alt text][image6]
![alt text][image7]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using LAB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Heres an example image:

![alt text][image8]
---

However this proved to be quite slow so one attempt to speed up the implementation was to reduce to number times the HOG features were calculated. This did speed up the pipeline a little however not too much. I didnt have enough time to further speed up the pipeline but some suggestions as given in the final section

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./writeup/video/final_video.mp4)

Some discussion on the final video:
1. When the cars first enter the image from the left it can take ~3 seconds for the pipeline to pick up the car. This is due to the implimented filter to remove false positives.
2. At ~8-9 seconds part of the yellow line is classified as a car. On debugging it classified multiple times in the frame and across multiple frame so unable to be removed through thresholding
3. The bounding boxes are often larger than the cars due to the window size, a fault of the sliding window technique. Solutions suggested below
4. When the white car passes the green road sign the boxes tries to include that. Could be something to do with some of the training images having simialr backgrounds leading to more of the bigger windows classifing
5. At ~28 seconds the white car is lost for about a second.
6. Once the blue car enters the image and once the two cars get close they are boxed together for about 10 seconds until they sufficiently split. This is always going to happen with this current pipeline. Suggestion below for improvements

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positions I created a heatmap and then thresholded that map to identify vehicle positions.  The thresholding acted as a filter of single frame false positives. This is where the high level of overlap assisted. By having more windows true positives would be selected a lot more creating a better split. This allowed for a slightly higher threshold of `threshold = 2`. This was initially higher but unfortunatly the white vehicle was lost as it drove further away sot he threshold was reduced. See the desicussion section for a potential fix for this

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. This could be improved by associating each 'blob' to a vehicle and tracking the vechile through multiple frame.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames, their corresponding heatmaps and their results bounding boxes

![alt text][image9]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There were quite a few problems that arose whilst doing thre project and potential future work:
1. Early on i had problems with classifing the vehicles in the windows pulled from the image. I played around with the classifier thinking that i had overfit my model. However cross validation showed that i still had a high accuracy. This is where is realised that having a lot more windows helped. By having more windows this just increased the possiblity of one being classified as positive
2. I had problems with the speed of the pipeline. It was no where near realtime (in fact as far from it as possible). I made severl attempts to improve the speed of the pipeline with little avail. One improvement I could try would to be to vectorise various areas of the pipeline. However i think that this approach itself is inherently slow. What is essentially manually stepping through numerous windows in probably fairly inefficient esspecially across all the frames. Ive seen that there a various other approached out there, such as YOLO, that use a Deep Learning approach that search the whole frame for the bounding boxes. I imagine this would be a more effecient approach running in near real time. Definately something to look at!
3. I had a still have problems with false positives. For this video its the yellow lines in the video. Several options are available such as feeding this into the dataset as Non-Vehicles to assist the classifier or improve the thresholding. But this raises the problem of fine tuning parameters to fit this solution not nessecarily a better general model. One solutions is to try and track the vehicles. Cars have more features that differentiate them from the surrounding feature other than just their appearance. Features such as their movement in the image, relative speed etc could be used to track the vehicles and then the positive classifications in new frames assigned to these already esstablished cars. Cars also enter then image from somewhere, often somewhere specific like the edges of the image or from the center of the image (in the case of the car catching up on vehicles). This could be used to intially assign a car object the could then be tracked through the image. This could also help in the case where a car blocks another car in the frame such as during about 28-38 seconds in the project video. The tracked vehicle never left the frame so must still be there there if could be safe to assume it is still there behind the other vehicle. Again, something to try going forward or to find out about
4. The bounding box often didn't fit tightly agaist the vehicle or picked out a small feature on the car. For the initially finding of the car size one solution could be to produce a larger window aorund the car based upon the intial centroid of the bounding box. Then trim the window to fit the car based upon the color of the car. This could be done through assumptions such color and contrast or runing a 'yolo' type approach across the image to find the tighter box. Once this tigher box has been found in following frames the bounding box and position could be asusmed similar and adjusted accordingly. If the cars relative speed is know (possible using image transforms similar to the previous project) then the bouding box could be scaled depending on whether the car is moving close or further away
5. There are definiatly areas where this pipeline would fail. As this is tried on one test video it is very tuned for it. Changes in anything, such as hills, lighting, anywhere the road isn't a nice relative straight peice of highway, could throw the pipeline off. Again this is where the ML approach would be advantageous as it would most likely do better at generalising (if handle conrrectly) so working in other and tricky environments.
