'''
Author: Qing Hong
Date: 2023-06-07 15:25:22
LastEditors: QingHong
LastEditTime: 2023-06-08 12:51:50
Description: file content
'''
import re
import os
import numpy as np
from PIL import ImageFilter

def pickle_write(path,data):
    import pickle
    with open(path, 'wb') as fh:
        pickle.dump(data, fh)

def create_lut_pkl(cube_file):
    with open(cube_file, 'r') as f:
        lines = f.read().split('\n')
    size_line = [line for line in lines if 'LUT_3D_SIZE' in line][0]
    size = int(size_line.split(' ')[-1])
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    data_lines = [line for line in lines if re.match(r"^\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$", line)]
    for i, line in enumerate(data_lines):
        r, g, b = list(map(float, line.split()))
        idx_b, idx_g, idx_r = np.unravel_index(i, (size, size, size))
        lut[idx_b, idx_g, idx_r] = [r, g, b]
    target_lut_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'lut',os.path.basename(cube_file).replace('.cube','.pkl'))
    final_data = {'size':size,'data':lut}
    pickle_write(target_lut_file,final_data)
    return final_data
# def create_lut_pkl(cube_file):
#     lut_file = read_txt(cube_file)
#     lut_data = None
#     lut_size = 65
#     tmp_flag = False
#     for file_data in lut_file:
#         if tmp_flag:
#             fs = re.findall(r'\b\d+\.\d+\b', file_data)
#             fs_data = np.array(fs).astype('float32')[None,...]
#             if lut_data is None:
#                 lut_data = fs_data
#             else:
#                 lut_data = np.concatenate((lut_data,fs_data))
#         if 'lut_3d_size' in file_data.lower():
#             tmp_flag = True
#             lut_size = int(re.findall('\d+',file_data)[-1])
#     assert lut_data is not None and lut_data.shape[0] == lut_size**3 ,'Lut file or data error!'
#     target_lut_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'lut',os.path.basename(cube_file).replace('.cube','.pkl'))
#     final_data = {'size':lut_size,'data':lut_data}
#     pickle_write(target_lut_file,final_data)
#     return final_data



def load_cube_file(lines, target_mode=None, cls=ImageFilter.Color3DLUT):
    """Loads 3D lookup table from .cube file format.

    :param lines: Filename or iterable list of strings with file content.
    :param target_mode: Image mode which should be after color transformation.
                        The default is None, which means mode doesn't change.
    :param cls: A class which handles the parsed file.
                Default is ``ImageFilter.Color3DLUT``.
    """
    from itertools import chain
    def isPath(f):
        return isinstance(f, (bytes, str))
    
    name, size = None, None
    channels = 3
    file = None

    if isPath(lines):
        file = lines = open(lines, 'rt')

    try:
        iterator = iter(lines)

        for i, line in enumerate(iterator, 1):
            line = line.strip()
            if line.startswith('TITLE "'):
                name = line.split('"')[1]
                continue
            if line.startswith('LUT_3D_SIZE '):
                size = [int(x) for x in line.split()[1:]]
                if len(size) == 1:
                    size = size[0]
                continue
            if line.startswith('CHANNELS '):
                channels = int(line.split()[1])
            if line.startswith('LUT_1D_SIZE '):
                raise ValueError("1D LUT cube files aren't supported")

            try:
                float(line.partition(' ')[0])
            except ValueError:
                pass
            else:
                # Data starts
                break

        if size is None:
            raise ValueError('No size found in the file')

        table = []
        for i, line in enumerate(chain([line], iterator), i):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                pixel = [float(x) for x in line.split()]
            except ValueError:
                raise ValueError("Not a number on line {}".format(i))
            if len(pixel) != channels:
                raise ValueError(
                    "Wrong number of colors on line {}".format(i))
            table.extend(pixel)
    finally:
        if file is not None:
            file.close()

    instance = cls(size, table, channels=channels,
                   target_mode=target_mode, _copy_table=False)
    if name is not None:
        instance.name = name
    return instance





import os, math, numpy as np, cv2
from pylut import LUT as lut


def generate_texture():
	# generate colors from (0,0,0) to (255,255,255), step 4
    step = 4
    gradients = int(256 / step) + 1
    lattice = []
    for b in range(0, gradients):
        for g in range(0, gradients):
            for r in range(0, gradients):
                blue = b * step if b < 64 else 255
                green = g * step if g < 64 else 255
                red = r * step if r < 64 else 255
                lattice.append((blue, green, red))  # channel order is BGR

    # because of possible jpeg compression, multiple blocks with gradation has to be made
    # furthurmore, each block will need to have a margin around
    # to avoid color corruption between abrupt color changes
    margin = 10
    blocksize = 65 + 2 * margin
    height = blocksize * 5
    width = blocksize * 13
    img = np.zeros((height, width, 3), np.int16)
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            # make grids of r by g
            blockx = math.floor(x / blocksize)  # 13 blocks in a row
            blocky = math.floor(y / blocksize)  # 5 blocks in a column
            residual_x = x % blocksize - margin
            if residual_x < 0:
                residual_x = 0
            elif residual_x > 64:
                residual_x = 64
            residual_y = y % blocksize - margin
            if residual_y < 0:
                residual_y = 0
            elif residual_y > 64:
                residual_y = 64
            count = blockx * 4225 + blocky * 4225 * 13 + residual_y * 65 + residual_x
            color = lattice[count]
            for z in range(0, 3):
                img[y, x, z] = color[z]  # channel order is BGR
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('texture.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def process_color(color):
    new = [1, 1, 1]
    new[0] = "{:.6f}".format(color[2] / 255)
    new[1] = "{:.6f}".format(color[1] / 255)
    new[2] = "{:.6f}".format(color[0] / 255)
    return new


def lut_from_texture(file):
    filename = os.path.basename(file)
    lutstring = list()
    img = cv2.imread(file, 1)  # channel order is BGR
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    if np.array(img).shape != (425, 1105, 3):
        print('Invalid texture image: ' + filename + " , process skipped.")
        return
    for i in range(0, 65):  # for each of 65 blocks
        posx = i % 13
        posy = math.floor(i / 13)
        startx = posx * 85
        starty = posy * 85
        block = img[starty:(starty + 84), startx:(startx + 84), :]
        core = block[10:75, 10:75, :]

        # read core area
        for y in range(0, 65):
            for x in range(0, 65):
                lutstring.append(process_color(core[y, x, :]))

    file = file[:-4]
    with open(file + '.cube', 'w') as outfile:
        outfile.write('TITLE "' + file + '"\n\n')
        outfile.write('LUT_3D_SIZE 65\n\n')
        for l in lutstring:
            for e in l:
                outfile.write(str(e) + ' ')
            outfile.write('\n')

    # resize to 32 cubesize and smooth, using modul pylut
    resize = lut.FromCubeFile(file + '.cube')
    resized = lut.Resize(resize, 32)
    resized.ToCubeFile(file + '.cube')
    print(filename, 'processed.')


# batch process
def batch_img2lut(targetdir):
    files = os.listdir(targetdir)
    for f in files:
        if os.path.splitext(f)[1].lower() == '.jpg':
            abspath = os.path.join(targetdir, f)
            lut_from_texture(abspath)