import numpy as np 
import os,sys
from file_utils import read,write
def mkdir(path):
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))

save_path = '/Users/qhong/Desktop/overlay_test/overlay_mv_ll_rr'
# mvs_1 = '/Users/qhong/Desktop/overlay_test/mmresult_l_r/greenbook_clip02_l/kousei-mask-fg-v0-230808300_monocular_object_mv0'
# mvs_2 = '/Users/qhong/Desktop/overlay_test/mmresult_l_r/greenbook_clip02_r/kousei-mask-fg-v0-230808300_monocular_object_mv0'
# mvs_ = jhelp_folder('/Users/qhong/Desktop/overlay_test/mmresult_llrr')
mvs_ = ['/Users/qhong/Desktop/overlay_test/mmresult_llrr/ll/greenbook_clip02/kousei-mask-fg-v0-230808300_monocular_object_mv0','/Users/qhong/Desktop/overlay_test/mmresult_llrr/lr/greenbook_clip02/kousei-mask-fg-v0-230808300_monocular_object_mv0','/Users/qhong/Desktop/overlay_test/mmresult_llrr/rl/greenbook_clip02/kousei-mask-fg-v0-230808300_monocular_object_mv0','/Users/qhong/Desktop/overlay_test/mmresult_llrr/rr/greenbook_clip02/kousei-mask-fg-v0-230808300_monocular_object_mv0']
mvs_ = [jhelp_file(k) for k in mvs_]
# mvs1 = jhelp_file(mvs_1)
# mvs2 = jhelp_file(mvs_2)

for i in range(len(mvs_[0])):
    mv = None
    for ii in range(len(mvs_)):
        mv_ = read(mvs_[ii][i],type='flo')
        if mv is None:
             mv = mv_
        else:
             mv = mv+mv_
    name = os.path.basename(mvs_[0][i])
    sp = os.path.join(save_path,name)
    write(sp,mv)

