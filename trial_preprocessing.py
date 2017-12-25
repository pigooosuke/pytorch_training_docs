from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()

landmarks_frame = pd.read_csv("faces/face_landmarks.csv")

n=65
img_name = landmarks_frame.ix[n ,0]
landmarks = landmarks_frame.ix[n,1:].as_matrix().astype("float")
landmarks = landmarks.reshape(-1,2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

def show_landmarks(image,landmarks):
    """show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1],s=10, marker=".",c="r")
    # plt.pause(0.001)

plt.figure()
show_landmarks(io.imread(os.path.join("faces/", img_name)),
    landmarks)
plt.show()


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset"""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file(string):Path to the csv file with annotations
            root_dir(string):Directory with all the images
            transform(callable,optional): Optional transform to be applied
                on a sample
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx,0])
        img = io.imread(img_name)
        landmarks = self.landmarks_frame.ix[idx,1:].as_matrix().astype("float")
        landmarks = landmarks.reshape(-1,2)
        sample = {"image":img, "landmarks":landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample

face_dataset = FaceLandmarksDataset(csv_file="faces/face_landmarks.csv",root_dir="faces/")
fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i,sample["image"].shape, sample["landmarks"].shape)

    ax = plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title("sample #{}".format(i))
    ax.axis("off")
    show_landmarks(**sample)
    if i==3:
        plt.show()
        break

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        image,landmarks = sample["image"], sample["landmarks"]
        h,w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h,new_w = self.output_size * h/w , self.output_size
            else:
                new_h,new_w = self.output_size , self.output_size * w/h
        else:
            new_h,new_w = self.output_size

        new_h,new_w = int(new_h),int(new_w)
        img = transform.resize(image,(new_h,new_w))
        landmarks = landmarks * [new_w/w, new_h/h]
        return {"image":img,"landmarks":landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size

    def __call__(self,sample):
        image, landmarks = sample["image"], sample["landmarks"]

        h,w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = image[top:top + new_h,
                        left:left + new_w]
        landmarks = landmarks - [left, top]
        return {"image":img,"landmarks":landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        image = image.transpose((2,0,1))
        return {"image":torch.from_numpy(image),
                "landmarks":torch.from_numpy(landmarks)}

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                                RandomCrop(224)])

fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale,crop,composed]):
    transformed_sample = tsfrm(sample)
    ax = plt.subplot(1,3,i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()
