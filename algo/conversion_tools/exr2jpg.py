'''
Author: Qing Hong
Date: 2023-04-19 15:46:46
LastEditors: QingHong
LastEditTime: 2023-04-19 15:47:01
Description: file content
'''
import os
import imageio
def convert_hdr_to_jpg(filepath):
    if not os.path.isfile(filepath):
        return False

    directory = os.path.dirname(filepath)
    filename, extension = os.path.splitext(filepath)
    if not extension.lower() in ['.hdr', '.hdri']:
        return False

    # imageio.plugins.freeimage.download() #DOWNLOAD IT
    image = imageio.imread(filepath, format='HDR-FI')
    output = os.path.join(directory, filename + '.jpg')
    imageio.imwrite(output, image)