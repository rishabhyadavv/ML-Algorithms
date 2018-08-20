#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:50:10 2018

@author: risyadav
"""

from sklearn.datasets import load_digits
digits = load_digits()
print(digits)

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print('Image Data Shape' , digits.data.shape)

# Print to show there are 1797 labels (integers from 0–9)
print('Label target Shape', digits.target.shape)

import numpy as np 
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))

for index, (image, label) in enumerate(zip(digits.data[0:4], digits.target[0:4])):
 plt.subplot(1, 4, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)
 
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

# Returns a NumPy Array
# Predict for One Observation (image)
logisticRegr.predict(x_test[0].reshape(1,-1))
#Make predictions on entire test data
logisticRegr.predict(x_test[0:10])
#Make predictions on entire test data
predictions = logisticRegr.predict(x_test)

# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)

#The confusion matrix below is not visually super informative or visually appealing.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics



cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')