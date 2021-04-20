from torch.utils.data import DataLoader
import torch
import pytorch_ssim
from astropy.io import fits
import numpy as np
import math
from math import log10

from torch.utils import data as data
from torchvision import transforms
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import os
from astropy.io import fits
from skimage.transform import resize
import numpy as np

IMG_EXTENSIONS = [".fits"]

def input_transform():
    return Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1500,))
    ])

def target_transform():
    return Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1500,))
    ])

def is_image_file(filename):
    """
    Helper Function to determine whether a file is an image file or not
    :param filename: the filename containing a possible image
    :return: True if file is image file, False otherwise
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    """
    Helper Function to make a dataset containing all images in a certain directory
    :param dir: the directory containing the dataset
    :return: images: list of image paths
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_fits_loader(file_name: str, img_size: tuple):
    file = fits.open(file_name)
    if np.all(file[0].data) == None:
        _data = file[1].data
    else:
        _data = file[0].data
#    file[1].verify('fix')
#    _data[_data != _data] = 0 
    _data = resize(_data, img_size)
    
    # _data = fits.get_data(file_name).resize(img_size)

    # add channels
    if len(_data.shape) < 3:
        _data = _data.reshape((*_data.shape,1))
    _target = _data
#    print(np.nanmax(_data))
    return _data, _target


class FITSDataset(data.Dataset):
    def __init__(self, data_path, input_transform, target_transform, img_size):
        self.data_path = data_path
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.img_size = img_size

        self.img_files = make_dataset(data_path)

    def __getitem__(self, index):
        _img, _target = default_fits_loader(self.img_files[index], self.img_size)
        _img = np.clip(_img,-1500,1500)
        _img = _img[1024:3072, 1024:3072]
#        print(np.nanmax(_img))
        _target = np.clip(_target,-1500,1500)
        _target = _target[1024:3072, 1024:3072]
#        print(np.nanmax(_target))
        _img = resize(_img, (1024, 1024),preserve_range=True,anti_aliasing=True)
#        print(np.nanmax(_img))
#        print(np.nanmin(_img))
        if self.input_transform:
            _img = self.input_transform(_img)
            _img[_img != _img] = -1
        if self.target_transform:
            _target = self.target_transform(_target)
            _target[_target != _target] = -1
        _data = (_img, _target)
        
        return _data

    def __len__(self):
        return len(self.img_files)

scale = 4
train_dataset = FITSDataset("/data/train", input_transform(),target_transform(),(1024*scale,1024*scale))
test_dataset = FITSDataset("/data/test", input_transform(),target_transform(),(1024*scale,1024*scale))
training_data_loader = data.DataLoader(train_dataset, num_workers=2, batch_size=4, shuffle=True)
testing_data_loader = data.DataLoader(test_dataset, num_workers=2, batch_size=4, shuffle=False)

def train():
    model.train()
    train_loss = 0
    for batch_num, (data, target) in enumerate(training_data_loader):
#        print(data.shape)
#        print(target.shape)
        optimizer.zero_grad()
        loss = criterion(model(data.float()), target.float())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar(batch_num, len(training_data_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

    print("    Average Loss: {:.4f}".format(train_loss / len(training_data_loader)))
    
def test():
    model.eval()
    avg_psnr = 0
    avg_ssim = 0

    with torch.no_grad():
        for batch_num, (data, target) in enumerate(testing_data_loader):
#            data, target = data.to(self.device), target.to(self.device)
            prediction = model(data.float())
            mse = criterion(prediction, target.float())
            psnr = 10 * log10(4 / mse.item())
            avg_psnr += psnr
            ssim = pytorch_ssim.ssim(prediction, target.float())
            avg_ssim += ssim 
            
            progress_bar(batch_num, len(testing_data_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

    print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("    Average SSIM: {:.4f} dB".format(avg_ssim / len(testing_data_loader)))
    
def main():
    model = Net(num_channels=1, upscale_factor=2, base_channel=64, num_residuals=4)
#DeepRCN
#model = Net(num_channels=1, base_channel=64, num_recursions=16, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#model.weight_init(mean=0.0, std=0.01)
    criterion = torch.nn.MSELoss()
    for epoch in range(1, 2):
    train()
    test() 




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

   

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
