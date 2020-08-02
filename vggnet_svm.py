import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from dataloader import get_loader
import numpy as np

from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score


batch_size = 256

trainloader = get_loader('../dataset/', 'train', batch_size = batch_size)

testloader = get_loader('../dataset/', 'val', batch_size = batch_size)

classes = ('angry', 'disgust', 'fear', 'happy',
           'sad', 'surprise', 'neutral')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.vgg13_bn(pretrained = True)
        for param in self.model.parameters():
            param.requires_grad = False
        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, 7)
        # print(self.model)
        # # create two list for feature and classifier blocks
        # self.fc1 = nn.Linear(1000, 256)
        # self.fc2 = nn.Linear(256, 7)
    def forward(self, x):
        x = self.model(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().cuda()
#net = nn.DataParallel(net).cuda()
labels_np = []
features_np = []
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs = inputs.type(torch.FloatTensor).cuda()
    outputs = net(inputs)
    features_cpu = outputs.cpu().data.numpy()
    labels_cpu = labels.data.numpy()
    print(features_cpu.shape)
    print(labels_cpu.shape)
    for i in range(labels.size(0)):
        features_np.append(features_cpu[i,:])
        labels_np.append(labels_cpu[i])

labels_train = np.asarray(labels_np)
features_train = np.asarray(features_np)
np.save('labels_vggnet', labels_train)
np.save('features_vggnet', features_train)

print('Train features Extracted!')

labels_np = []
features_np = []
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = inputs.type(torch.FloatTensor).cuda()
    outputs = net(inputs)
    features_cpu = outputs.cpu().data.numpy()
    labels_cpu = labels.data.numpy()
    print(features_cpu.shape)
    print(labels_cpu.shape)
    for i in range(labels.size(0)):
        features_np.append(features_cpu[i,:])
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
