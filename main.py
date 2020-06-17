
#
import glob
import os
import argparse

from PIL import Image
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import  GroupKFold

from models import *
from bilinear_layers import *

import utils


class oaDataset(Dataset):
    def __init__(self, root_path, transform):
        # store filenames
        self.roi_root = root_path
        self.flist = glob.glob(os.path.join(self.roi_root, "**"))
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.flist)

    def __getitem__(self, idx):
        fname = self.flist[idx]
        image, min_jsw, jsw_des, fjsw, kl, jsize = np.load(fname, allow_pickle=True)  # PIL image

        if kl > 1:
            grade =1
        else:
            grade = 0

        image = Image.fromarray(image.astype('uint8'), 'L')
        image = self.transform(image)

        jsw = np.array(jsw_des)
        # noise = np.random.normal(0, 1, 221)
        # jsw = jsw + noise
        jsw = torch.from_numpy(jsw).float()

        return image, jsw, grade


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--oai_assesment', default='/../DATA/OAI/kXR_SQ_BU00_SAS/kxr_sq_bu00.sas7bdat')
    parser.add_argument('--model', default=combined)
    args = parser.parse_args()

    SEED = 42
    MAX_EPOCH = 100

    np.random.seed(SEED)
    torch.manual_seed(SEED);
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    cv = GroupKFold(n_splits=5)

    train_transforms = transforms.Compose([
        transforms.Resize(size=(56,56)),
        transforms.RandomCrop(size= (48,48)),
        transforms.ToTensor(),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(size=(48,48)),
        transforms.ToTensor(),
    ])

    plt.figure(figsize=(10, 10))

    train_root = '../DATA/CROPS/OAI_00m_tm_fjsw_standardized/'
    test_root = '../DATA/CROPS/MOST_00m_tm_fjsw_standardized/'

    train_dataloader = DataLoader(oaDataset(train_root, train_transforms),
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)
    test_dataloader = DataLoader(oaDataset(test_root, eval_transforms),
                                 batch_size=64,
                                 num_workers=8,
                                 drop_last=True)

    net = args.model().to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True)

    best_acc = 0.0
    best_epoch = None

    fpr = []
    tpr = []

    for epoch in range(MAX_EPOCH):

        optimizer,lr = utils.adjust_learning_rate(optimizer,epoch,0.00001, 0.01, 8)

        # Training
        train_loss = utils.train_epoch(epoch, net, optimizer, train_dataloader, criterion)
        # Validating
        val_loss, preds, truth = utils.validate_epoch(net, test_dataloader, criterion)

        auc_val = roc_auc_score(truth, preds)
        print(epoch + 1, train_loss, val_loss, auc_val)

        if auc_val > best_acc:
            best_acc = auc_val
            best_epoch = epoch
            fpr, tpr, thresholds = roc_curve(truth, preds)

    plt.plot(fpr, tpr, lw=2, alpha=0.8,
             label='ROC (AUC = %0.2f)' % (best_acc))
    plt.show()

    print("Best AUC:", best_acc)
