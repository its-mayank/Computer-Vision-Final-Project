import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from dataloader import get_loader


batch_size = 64

trainloader = get_loader('../dataset/', 'train', batch_size = batch_size)

testloader = get_loader('../dataset/', 'val', batch_size = batch_size)

classes = ('angry', 'disgust', 'fear', 'happy',
           'sad', 'surprise', 'neutral')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained = True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 7)
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

epochs = 50
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    PATH = './fer_resnet18.pth'
    torch.save(net, PATH)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels.type(torch.LongTensor).cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

print('Finished Training')
