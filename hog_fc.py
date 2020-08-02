import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from dataloader import get_loader
from scipy.misc import toimage
from skimage import color
from skimage.feature import hog

batch_size = 256

trainloader = get_loader('../dataset/', 'train', batch_size = batch_size)

testloader = get_loader('../dataset/', 'val', batch_size = batch_size)

classes = ('angry', 'disgust', 'fear', 'happy',
		   'sad', 'surprise', 'neutral')

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(1296, 256)
		self.fc2 = nn.Linear(256, 7)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochs = 50
for epoch in range(epochs):  # loop over the dataset multiple times
	running_loss = 0.0
	batch = 0
	for data in trainloader:
		inputs, labels = data
		numpy_input = inputs.data.numpy()
		inputs_new = []
		for i in range(inputs.size(0)):
			img = numpy_input[i, :, :, :]
			img = np.reshape(img, (img.shape[1], img.shape[2], img.shape[0]))
			img_gray = np.asarray(color.rgb2gray(img))
			features, hog_image = hog(img_gray, block_norm= 'L2', visualize=True)
			
			inputs_new.append(features)
		inputs_new = (torch.from_numpy(np.asarray(inputs_new))).type(torch.FloatTensor)
		inputs = inputs_new.cuda()
		labels = labels.cuda()
		# zero the parameter gradients
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		batch+=1
		running_loss += loss.item()
		if batch % 20 == 19:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 20))
			running_loss = 0.0

	PATH = './fer_hog_fc.pth'
	torch.save(net, PATH)

	correct = 0
	total = 0
	print('Going in Validation Phase!')
	with torch.no_grad():
		for data in testloader:
			inputs, labels = data
			#Compute the LBP features here
			numpy_input = inputs.data.numpy()
			inputs_new = []
			for i in range(inputs.size(0)):
				img = numpy_input[i, :, :, :]
				img = np.reshape(img, (img.shape[1], img.shape[2], img.shape[0]))
				img_gray = np.asarray(color.rgb2gray(img))
				features, hog_image = hog(img_gray, block_norm= 'L2', visualize=True)
				inputs_new.append(features)
			inputs_new = (torch.from_numpy(np.asarray(inputs_new))).type(torch.FloatTensor)
			# print(type(inputs_new))
			#print(inputs_new.shape)
			inputs = inputs_new.cuda()
			labels = labels.cuda()
			outputs = net(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			#print(predicted.shape, labels.shape)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))

print('Finished Training')
