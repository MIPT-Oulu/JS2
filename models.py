from __future__ import division
from bilinear_layers import *
from models_lpf import *


class plain_model_3layers(nn.Module):
    def __init__(self):
        super(plain_model_3layers, self).__init__()
        # we define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(in_features=128 * 5 * 5, out_features=128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.dropout_rate = 0.5
        self.drop_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, s, jsw):
        # we apply the convolution layers, followed by batch normalisation,
        # maxpool and relu x 3

        s = self.bn1(self.conv1(s))
        s = F.relu(F.max_pool2d(s, 2))

        s = self.bn2(self.conv2(s))
        s = F.relu(F.max_pool2d(s, 2))

        s = self.bn3(self.conv3(s))
        s = F.relu(F.max_pool2d(s, 2))

        # flatten the output for each image
        s = s.view(-1, 5 * 5 * 128)
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                      p=self.dropout_rate, training=self.training)  # batch_size x 128
        s = self.fc2(s)
        return s


class plain_model_4layers(nn.Module):
    def __init__(self):
        super(plain_model_4layers, self).__init__()
        # we define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(in_features=256 * 3 * 3, out_features=128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.dropout_rate = 0.5
        self.drop_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, s, jsw):
        # we apply the convolution layers, followed by batch normalisation,
        # maxpool and relu x 3

        s = self.bn1(self.conv1(s))
        s = F.relu(F.max_pool2d(s, 2))

        s = self.bn2(self.conv2(s))
        s = F.relu(F.max_pool2d(s, 2))

        s = self.bn3(self.conv3(s))
        s = F.relu(F.max_pool2d(s, 2))

        s = self.bn4(self.conv4(s))
        s = F.relu(F.max_pool2d(s, 2))

        # flatten the output for each image
        s = s.view(-1, 3 * 3 * 256)
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                      p=self.dropout_rate, training=self.training)  # batch_size x 128
        s = self.fc2(s)
        return s


class improved_bcnn(nn.Module):
    def __init__(self):
        super(improved_bcnn, self).__init__()
        # we define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(in_features=128 * 128, out_features=128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.dropout_rate = 0.5

        self.matrix_sqrt = matrix_sqrt.apply
        self.sign_sqrt = sign_sqrt.apply
        self.classifier = torch.nn.Linear(
            in_features=128 * 128, out_features=1, bias=True)
        # torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, s, jsw):
        # we apply the convolution layers, followed by batch normalisation,
        # maxpool and relu x 3

        N = s.size()[0]  # batch size
        s = self.bn1(self.conv1(s))
        s = F.relu(F.max_pool2d(s, 2))
        s = self.bn2(self.conv2(s))
        s = F.relu(F.max_pool2d(s, 2))
        s = self.bn3(self.conv3(s))
        s = F.relu(F.max_pool2d(s, 2))

        s = torch.reshape(s, (N, 128, 6 * 6))
        s = torch.bmm(s, torch.transpose(s, 1, 2)) / (6 * 6)
        assert s.size() == (N, 128, 128)
        s = self.matrix_sqrt(s + 1e-8)
        s = self.sign_sqrt(s + 1e-8)
        s = torch.reshape(s, (N, 128 * 128))

        s = torch.nn.functional.normalize(s)
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                      p=self.dropout_rate, training=self.training)  
        s = self.fc2(s)

        return s


class antialised_cnn(nn.Module):

    def __init__(self):
        super(antialised_cnn, self).__init__()
        # we define convolutional layers

        filter_size = 1

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.d1 = Downsample(filt_size=filter_size, stride=2, channels=32, pad_off=-1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.d2 = Downsample(filt_size=filter_size, stride=2, channels=64, pad_off=-1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.d3 = Downsample(filt_size=filter_size, stride=2, channels=128, pad_off=-1)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(in_features=128 * 6 * 6, out_features=128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.dropout_rate = 0.5
        self.drop_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, s, jsw):
        # we apply the convolution layers, followed by batch normalisation,
        # maxpool and relu x 3

        N = s.size()[0]  # batch size
        s = self.bn1(self.conv1(s))
        s = self.d1(F.relu(s))

        s = self.bn2(self.conv2(s))
        s = self.d2(F.relu(s))

        s = self.bn3(self.conv3(s))
        s = self.d3(F.relu(s))

        s = s.contiguous().view(s.size(0), 128 * 6 * 6)
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                      p=self.dropout_rate, training=self.training)  
        s = self.fc2(s)
        return s


class jsw(nn.Module):
    def __init__(self):
        super(jsw, self).__init__()
        self.fc1 = nn.Linear(in_features=221, out_features=128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, s, jsw):
        # we apply the convolution layers, followed by batch normalisation,
        # maxpool and relu x 3
        input = s
        N = jsw.size()[0]  # batch size
        s = jsw.view(-1, 221)  
        s = F.relu(self.fcbn1(self.fc1(s))) 
        s = self.fc2(s)  
        return s


class combined(nn.Module):
    def __init__(self):
        super(combined, self).__init__()
        # we define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(in_features=128 * 6 * 6 + 221, out_features=128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.dropout_rate = 0.3
        self.drop_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, s, jsw):
        # we apply the convolution layers, followed by batch normalisation,
        # maxpool and relu x 3
        input = s
        N = s.size()[0]  # batch size
        s = self.bn1(self.conv1(s))  
        s = F.relu(F.max_pool2d(s, 2))  

        s = self.bn2(self.conv2(s))  
        s = F.relu(F.max_pool2d(s, 2)) 

        s = self.bn3(self.conv3(s)) 
        s = F.relu(F.max_pool2d(s, 2)) 

        # flatten the output for each image
        s = s.view(-1, 6 * 6 * 128)  
        s = torch.cat((s, jsw), dim=1)

        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                      p=self.dropout_rate, training=self.training)  
        s = self.fc2(s) 
        return s


class morphology(nn.Module):
    def __init__(self):
        super(morphology, self).__init__()
        self.fc1 = nn.Linear(in_features=221, out_features=128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, s, jsw):

        s = jsw.view(-1, 221)
        s = F.relu(self.fcbn1(self.fc1(s)))
        s = self.fc2(s)
        return s
