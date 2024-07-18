'''
Author: Qing Hong
Date: 2024-03-06 10:30:14
LastEditors: QingHong
LastEditTime: 2024-03-29 12:45:43
Description: file content
'''
import PyOpenColorIO as ocio
import os
class Color_transform:
    def __init__(self,src='lin_rec709',dst='acescg'):
        self.src = src
        self.dst = dst
        self.transform = None
    def init(self,):
        try:
            ocio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.ocio')
            config = ocio.Config.CreateFromFile(ocio_path)
            self.transform = config.getProcessor(srcColorSpaceName=self.src,dstColorSpaceName=self.dst).getDefaultCPUProcessor()
        except:
            print('wrong color space')
    def apply(self, image):
        if self.transform is None:
            self.init()
        self.transform.applyRGB(image)
        return image
    


