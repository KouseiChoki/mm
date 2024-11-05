import torchvision
#import torchvision.transforms as TF
import torchvision.datasets as datasets
from torchvision import transforms

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import os
import glob
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
 

convert_tensor = transforms.ToTensor()

def mask_rgb(rgbImage, mask):
    # create 3 channel of the mask
    mask3chnl = torch.cat([mask,mask,mask], dim=0)
    # mask is already 3 channel???
    #return rgbImage*mask3chnl+0.5*(1-mask3chnl)
    # let's put hole in as black
    return rgbImage*mask3chnl

def get_dataset(dataset_name='MNIST',imgSize=(32,32)):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(imgSize, 
                                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC, 
                                          antialias=True),
            torchvision.transforms.RandomHorizontalFlip(),
#             torchvision.transforms.Normalize(MEAN, STD),
            torchvision.transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
        ]
    )
     
    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-10":    
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Flowers":
        dataset = datasets.ImageFolder(root="/Volumes/neil_SSD/datasets/flowers", transform=transforms)
 
    return dataset
 
def inverse_transform(tensors,maxVal=255.0):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * maxVal

class inpaint_train_dataset(Dataset):
    def __init__(self, args, imgFolder, maskFolder, imgTransform=None, maskTransform=None):
        self.folderPath = imgFolder
        #self.fileList = glob.glob(os.path.join(imgFolder, '*'))
        #self.fileList = glob.glob(imgFolder, '**',recursive=True)
        self.fileList = []
        for i in range(len(imgFolder)):
            tempfileList = glob.glob(os.path.join(imgFolder[i], '**/*.*'),recursive=True)
            self.fileList += tempfileList
            print(f'[Folder: {imgFolder[i]}] Len: {len(tempfileList)} | cum len: {len(self.fileList)}')
        #self.fileList = glob.glob(os.path.join(imgFolder, '**/*.*'),recursive=True)
        #folderList = glob.glob(os.path.join(imgFolder, '*'))
        #if args.imgSubFolders > 0:
        #    for i in len(folderList):
        #        if i==0:
        #            self.fileList = glob.glob(os.path.join(folderList[i], '*'))
        #        else:
        #            self.fileList += glob.glob(os.path.join(folderList[i], '*'))
        #else:
        #    self.fileList = folderList
        self.maskPath = maskFolder
        self.maskFileList = glob.glob(os.path.join(maskFolder, '*'))
        self.transform = imgTransform
        self.maskTransform = maskTransform
        self.maskIsPV = args.maskIsPV
        self.imgIs16 = args.imgIs16
        self.maskIs16 = args.maskIs16
        self.maskG = args.inMaskG
        
    def __getitem__(self, index):
        imgPath = self.fileList[index]
        imgDirFile = os.path.split(imgPath) # [path to the file, filename.ext]
        imgName = os.path.splitext(imgDirFile[1])[0] # just the name, drop the extension
        tempPic = cv2.imread(imgPath,6)
        tempPic = tempPic.astype('float32')
        if self.imgIs16:
            tempPic = tempPic/65535
        else:
            tempPic = tempPic/255
        if len(tempPic.shape) == 3:
            #tempPic = cv2.cvtColor(tempPic, cv2.COLOR_BGR2YCrCb)
            image = convert_tensor(tempPic)
            #image = ycrcb2yuv(image,self.cGain) # so that 0 color has CbCr = 0
        else:
            temp = convert_tensor(tempPic)
            image = torch.concat([temp,temp,temp],0)
        if(self.transform):
        #    transform_choice = random.choice([0,1,1,2,2,3,3]) # reduce number of times with black border
        #    image = self.transform[transform_choice](image)
            image = self.transform(image)
        maskLen = len(self.maskFileList)
        maskPath = self.maskFileList[random.randrange(0,maskLen)]
        tempMask = cv2.imread(maskPath,6)
        tempMask = tempMask.astype('float32')
        if self.maskIs16:
            tempMask = tempMask/65535
        else:
            tempMask = tempMask/255
        if len(tempMask.shape) == 3:
            tempMask = cv2.cvtColor(tempMask, cv2.COLOR_BGR2GRAY)
        mask = convert_tensor(tempMask)
        if not self.maskIsPV: # then invert mask so that any pad is considered the mask
            mask = 1-mask
        if(self.transform):
        #    transform_choice = random.choice([0,1,2,3])
        #    mask = self.maskTransform[transform_choice](mask)
            mask = self.maskTransform(mask)
        #with torch.no_grad(): # don't include this in the calculations
        #if self.maskIsPV:
        mask = 1-mask # then invert so that mask is 1
        mask = torch.clip(mask*self.maskG,0,1)
        inVpix = 1-mask
        inImage = mask_rgb(image, inVpix)
        #inImage = image # if mask is not 0/1...but input mask is 0/1 and scaling * 512 is >1 if any input > 0 
        # construct the input dataset
        #inputStack = torch.cat([inImage,inVpix,mask],dim=0)
        inputStack = torch.cat([inImage,inVpix],dim=0)
        # no offset, only 0-1
        #inputStack = (inputStack - 0.5)*2
        #image = (image-0.5)*2

        return image, inputStack, imgName
             
    def __len__(self):
        return len(self.fileList)

class inpaint_mask_dataset(Dataset):
    def __init__(self, args, imgFolder, maskFolder, imgTransform=None, maskTransform=None):
        self.folderPath = imgFolder
        self.maskPath = maskFolder
        if args.getRandMask:
            self.fileList = []
            for i in range(len(imgFolder)):
                tempfileList = glob.glob(os.path.join(imgFolder[i], '**/*.*'),recursive=True)
                self.fileList += tempfileList
                print(f'[Folder: {imgFolder[i]}] Len: {len(tempfileList)} | cum len: {len(self.fileList)}')
            self.maskFileList = glob.glob(os.path.join(maskFolder, '*'))
        else:
            self.fileList = sorted(glob.glob(os.path.join(imgFolder[0], '*')))
            self.maskFileList = sorted(glob.glob(os.path.join(maskFolder, '*')))
        self.transform = imgTransform
        self.maskTransform = maskTransform
        self.maskIsPV = args.maskIsPV
        self.imgIs16 = args.imgIs16
        self.maskIs16 = args.maskIs16
        #self.maskG = args.inMaskG
        self.chromaKey = args.chromaKey
        self.getRandMask = args.getRandMask
        
    def __getitem__(self, index):
        imgPath = self.fileList[index]
        imgDirFile = os.path.split(imgPath) # [path to the file, filename.ext]
        imgName = os.path.splitext(imgDirFile[1])
        tempPic = cv2.imread(imgPath,6)
        tempPic = tempPic.astype('float32')
        if imgName[1] != '.exr':
            if self.imgIs16:
                tempPic = tempPic/65535
            else:
                tempPic = tempPic/255
        if len(tempPic.shape) == 3:
            #tempPic = cv2.cvtColor(tempPic, cv2.COLOR_BGR2YCrCb)
            image = convert_tensor(tempPic)
            #image = ycrcb2yuv(image,self.cGain) # so that 0 color has CbCr = 0
        else:
            temp = convert_tensor(tempPic)
            image = torch.concat([temp,temp,temp],0)
        #maskLen = len(self.maskFileList)
        if self.chromaKey is not None: # has to be done before any transforms and result is a pixel valid
            tempMask = torch.zeros_like(image)
            for i in range(3):
                #tempMask[i,:,:] = image[i,:,:].masked_fill(image[i,:,:] == self.chromaKey[i], 2) # so mask value is 2x max image value
                tempMask[i,:,:] = tempMask[i,:,:].masked_fill(image[i,:,:] == self.chromaKey[i], 2) # so mask value is 2x max image value
            mask = tempMask[0:1,:,:] + tempMask[1:2,:,:] + tempMask[2:3,:,:]
            mask = F.threshold(mask,5.5,0)/6 # two channel match max value = 2+2+1 = 5, so 5.5 threshold
            mask = 1-mask #expand mask and change to
            #mask = 1-F.max_pool2d(mask,3,stride=1,padding=1) #expand mask and change to
            # mask is actually pixel valid so that during transform any padding = 0 is an area to inpaint
        else:
            if self.getRandMask:
                maskLen = len(self.maskFileList)
                maskPath = self.maskFileList[random.randrange(0,maskLen)]
            else:
                maskPath = self.maskFileList[index]
            maskDirFile = os.path.split(maskPath) # [path to the file, filename.ext]
            maskName = os.path.splitext(maskDirFile[1])
            tempMask = cv2.imread(maskPath,6)
            tempMask = tempMask.astype('float32')
            if maskName[1] != '.exr':
                if self.maskIs16:
                    tempMask = tempMask/65535
                else:
                    tempMask = tempMask/255
            if len(tempMask.shape) == 3:
                tempMask = cv2.cvtColor(tempMask, cv2.COLOR_BGR2GRAY)
            mask = convert_tensor(tempMask)
            if not self.maskIsPV: # then invert mask so that any pad is considered the mask
                mask = 1-mask
        if(self.maskTransform):
        #    transform_choice = random.choice([0,1,2,3])
        #    mask = self.maskTransform[transform_choice](mask)
            mask = self.maskTransform(mask)
        #with torch.no_grad(): # don't include this in the calculations
        #if self.maskIsPV:
        mask = 1-mask # then invert so that mask is 1
        #mask = torch.clip(mask*self.maskG,0,1)
        mask = mask.clamp(0,1)
        mask = torch.ceil(mask-.01)
        if(self.transform):
        #    transform_choice = random.choice([0,1,1,2,2,3,3]) # reduce number of times with black border
        #    image = self.transform[transform_choice](image)
            image = self.transform(image)
        #image = torch.sqrt(image)
        inVpix = 1-mask
        inImage = mask_rgb(image, inVpix)
        # construct the input dataset
        #inputStack = torch.cat([inImage,inVpix,mask],dim=0)
        inputStack = torch.cat([inImage,inVpix],dim=0)
        # no offset, only 0-1
        #inputStack = (inputStack - 0.5)*2
        #image = (image-0.5)*2

        return image, inputStack, imgName[0]
             
    def __len__(self):
        return len(self.fileList)

class inpaint_unreal_dataset(Dataset):
    def __init__(self, args, imgFolder, maskFolder, imgTransform=None, maskTransform=None):
        self.folderPath = imgFolder
        self.fileList = sorted(glob.glob(os.path.join(imgFolder[0], '**/image/*.*'),recursive=True)) # to load all the files in the subfolders
        #self.maskPath = maskFolder
        #self.maskFileList = sorted(glob.glob(os.path.join(maskFolder, '*')))
        self.transform = imgTransform
        #self.maskTransform = maskTransform
        #self.maskIsPV = args.maskIsPV
        self.imgIs16 = args.imgIs16
        self.maskIs16 = args.maskIs16
        #self.maskG = args.inMaskG
        
    def __getitem__(self, index):
        imgPath = self.fileList[index]
        imgDirFile = os.path.split(imgPath) # [path to the file, filename.ext]
        imgName = os.path.splitext(imgDirFile[1])
        tempPic = cv2.imread(imgPath,6)
        tempPic = tempPic.astype('float32')
        if imgName[1] != '.exr':
            if self.imgIs16:
                tempPic = tempPic/65535
            else:
                tempPic = tempPic/255
        if len(tempPic.shape) == 3:
            #tempPic = cv2.cvtColor(tempPic, cv2.COLOR_BGR2YCrCb)
            image = convert_tensor(tempPic)
            #image = ycrcb2yuv(image,self.cGain) # so that 0 color has CbCr = 0
        else:
            temp = convert_tensor(tempPic)
            image = torch.concat([temp,temp,temp],0)
        if(self.transform):
        #    transform_choice = random.choice([0,1,1,2,2,3,3]) # reduce number of times with black border
        #    image = self.transform[transform_choice](image)
            image = self.transform(image)
        #maskLen = len(self.maskFileList)
        #maskPath = self.maskFileList[index]
        #maskDirFile = os.path.split(maskPath) # [path to the file, filename.ext]
        #maskName = os.path.splitext(maskDirFile[1])
        #tempMask = cv2.imread(maskPath,6)
        #tempMask = tempMask.astype('float32')
        #if maskName[1] != '.exr':
        #    if self.maskIs16:
        #        tempMask = tempMask/65535
        #    else:
        #        tempMask = tempMask/255
        #if len(tempMask.shape) == 3:
        #    tempMask = cv2.cvtColor(tempMask, cv2.COLOR_BGR2GRAY)
        #mask = convert_tensor(tempMask)
        #if not self.maskIsPV: # then invert mask so that any pad is considered the mask
        #    mask = 1-mask
        ##if(self.transform):
        ##    transform_choice = random.choice([0,1,2,3])
        ##    mask = self.maskTransform[transform_choice](mask)
        #    mask = self.maskTransform(mask)
        ##with torch.no_grad(): # don't include this in the calculations
        ##if self.maskIsPV:
        #mask = 1-mask # then invert so that mask is 1
        #mask = torch.clip(mask*self.maskG,0,1)
        #inVpix = 1-mask
        #inImage = mask_rgb(image, inVpix)
        inImage = image
        inputStack = inImage
        #inImage = image # if mask is not 0/1...but input mask is 0/1 and scaling * 512 is >1 if any input > 0 
        # construct the input dataset
        #inputStack = torch.cat([inImage,inVpix,mask],dim=0)
        #inputStack = torch.cat([inImage,inVpix],dim=0)
        # no offset, only 0-1
        #inputStack = (inputStack - 0.5)*2
        #image = (image-0.5)*2

        return image, inputStack, imgName[0]
             
    def __len__(self):
        return len(self.fileList)

class inpaint_random_dataset(Dataset):
    def __init__(self, args, imgFolder, maskFolder, imgTransform=None, maskTransform=None):
        self.folderPath = imgFolder
        #self.fileList = glob.glob(os.path.join(imgFolder, '*'))
        #self.fileList = glob.glob(imgFolder, '**',recursive=True)
        self.fileList = []
        for i in range(len(imgFolder)):
            tempfileList = glob.glob(os.path.join(imgFolder[i], '**/*.*'),recursive=True)
            self.fileList += tempfileList
            print(f'[Folder: {imgFolder[i]}] Len: {len(tempfileList)} | cum len: {len(self.fileList)}')
        #self.fileList = glob.glob(os.path.join(imgFolder, '**/*.*'),recursive=True)
        #folderList = glob.glob(os.path.join(imgFolder, '*'))
        #if args.imgSubFolders > 0:
        #    for i in len(folderList):
        #        if i==0:
        #            self.fileList = glob.glob(os.path.join(folderList[i], '*'))
        #        else:
        #            self.fileList += glob.glob(os.path.join(folderList[i], '*'))
        #else:
        #    self.fileList = folderList
        self.maskPath = maskFolder
        self.maskFileList = glob.glob(os.path.join(maskFolder, '*'))
        self.transform = imgTransform
        self.maskTransform = maskTransform
        self.maskIsPV = args.maskIsPV
        self.imgIs16 = args.imgIs16
        self.maskIs16 = args.maskIs16
        self.maskG = args.inMaskG
        
    def __getitem__(self, index):
        imgPath = self.fileList[index]
        imgDirFile = os.path.split(imgPath) # [path to the file, filename.ext]
        imgName = os.path.splitext(imgDirFile[1])[0] # just the name, drop the extension
        tempPic = cv2.imread(imgPath,6)
        tempPic = tempPic.astype('float32')
        if self.imgIs16:
            tempPic = tempPic/65535
        else:
            tempPic = tempPic/255
        if len(tempPic.shape) == 3:
            #tempPic = cv2.cvtColor(tempPic, cv2.COLOR_BGR2YCrCb)
            image = convert_tensor(tempPic)
            #image = ycrcb2yuv(image,self.cGain) # so that 0 color has CbCr = 0
        else:
            temp = convert_tensor(tempPic)
            image = torch.concat([temp,temp,temp],0)
        if(self.transform):
        #    transform_choice = random.choice([0,1,1,2,2,3,3]) # reduce number of times with black border
        #    image = self.transform[transform_choice](image)
            image = self.transform(image)
        maskLen = len(self.maskFileList)
        maskPath = self.maskFileList[random.randrange(0,maskLen)]
        tempMask = cv2.imread(maskPath,6)
        tempMask = tempMask.astype('float32')
        if self.maskIs16:
            tempMask = tempMask/65535
        else:
            tempMask = tempMask/255
        if len(tempMask.shape) == 3:
            tempMask = cv2.cvtColor(tempMask, cv2.COLOR_BGR2GRAY)
        mask = convert_tensor(tempMask)
        if not self.maskIsPV: # then invert mask so that any pad is considered the mask
            mask = 1-mask
        if(self.transform):
        #    transform_choice = random.choice([0,1,2,3])
        #    mask = self.maskTransform[transform_choice](mask)
            mask = self.maskTransform(mask)
        #with torch.no_grad(): # don't include this in the calculations
        #if self.maskIsPV:
        mask = 1-mask # then invert so that mask is 1
        mask = torch.clip(mask*self.maskG,0,1)
        inVpix = 1-mask
        inImage = mask_rgb(image, inVpix)
        #inImage = image # if mask is not 0/1...but input mask is 0/1 and scaling * 512 is >1 if any input > 0 
        # construct the input dataset
        #inputStack = torch.cat([inImage,inVpix,mask],dim=0)
        inputStack = torch.cat([inImage,inVpix],dim=0)
        # no offset, only 0-1
        #inputStack = (inputStack - 0.5)*2
        #image = (image-0.5)*2

        return image, inputStack, imgName
             
    def __len__(self):
        return len(self.fileList)
