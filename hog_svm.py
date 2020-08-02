import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_loader
import numpy as np
from skimage.feature import hog
from sklearn import svm
from skimage import color
from sklearn.metrics import classification_report,accuracy_score


batch_size = 256

trainloader = get_loader('../dataset/', 'train', batch_size = batch_size)

testloader = get_loader('../dataset/', 'val', batch_size = batch_size)

classes = ('angry', 'disgust', 'fear', 'happy',
           'sad', 'surprise', 'neutral')

labels_np = []
features_np = []
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs = inputs.data.numpy()
    labels_cpu = labels.data.numpy()
    for i in range(labels.size(0)):
        img = np.reshape(inputs[i, :, :, :],(48, 48, 3))
        img_gray = np.asarray(color.rgb2gray(img))
        fd,hog_image = hog(img_gray, block_norm= 'L2', visualize=True)
        features_np.append(fd)
        labels_np.append(labels_cpu[i])

labels_train = np.asarray(labels_np)
features_train = np.asarray(features_np)
print(labels_train.shape)
print(features_train.shape)
np.save('labels_HOG', labels_train)
np.save('features_HOG', features_train)

print('Train features Extracted!')

labels_np = []
features_np = []
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = inputs.data.numpy()
    labels_cpu = labels.data.numpy()
    for i in range(labels.size(0)):
        img = np.reshape(inputs[i, :, :, :],(48, 48, 3))
        img_gray = np.asarray(color.rgb2gray(img))
        fd,hog_image = hog(img_gray, block_norm= 'L2', visualize=True)
        features_np.append(fd)
        labels_np.append(labels_cpu[i])

labels_test = np.asarray(labels_np)
features_test = np.asarray(features_np)

print('Test features Extracted!')

clf = svm.SVC()
print('SVM Initialized!')
clf.fit(features_train, labels_train)
print('SVM Trained!')
prediction = clf.predict(features_test)

print("Accuracy: "+str(accuracy_score(labels_test, prediction)))
print('\n')
print(classification_report(labels_test, prediction))
