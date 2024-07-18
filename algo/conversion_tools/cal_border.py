import os,sys
from myutil import jhelp,read,write,mkdir,get_board_length

if __name__ == '__main__':
    assert len(sys.argv)>1,'usage python border_length.py image'
    image_ = sys.argv[1]
    image = read(image_,type='image')
    print(get_board_length(image))