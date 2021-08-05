# IRIS-Analyzer

Given the widespread use of classical texture descriptors (texture features that are ex-tracted from the iris) for iris recognition, including the Gabor phase-quadrant features(forextracting these features) , it is instructive to take a step back and answer the followingquestion: how do we know that these image processing feature descriptors proposed in theliterature are actually the best representations for the iris?  Furthermore, can we achievebetter performance (compared to the Gabor-based procedure) by designing a better fea-ture representation scheme that can perhaps attain the upper bound on iris recognitionaccuracy with low computational complexity?3
Deep learning techniques are one of the best ways to solve the above problem as itincreases the accuracy,by using improved feature extraction techniques, that is primarilydata-driven.Use of Convolution Neural Networks (CNN) models to train, classify and test images,gives better accuracy than previous image processing techniques.  Images to be used in theCNN should be preprocessed:  cropped, converted to numpy arrays and normalised(imagepixels divided by 255 to make the pixels in a range of 0-1 for easier and faster processing).
