
import os,sys
import re

def jhelp(c):
	return [os.path.join(c,i) for i in list(filter(lambda x:x[0]!='.',sorted(os.listdir(c))))]
def jhelp_folder(c):
    return list(filter(lambda x:os.path.isdir(x),jhelp(c)))
def jhelp_file(c):
    return list(filter(lambda x:not os.path.isdir(x),jhelp(c)))
root = sys.argv[1]
# root = '/Volumes/dept/ITSync2021/7dayexpire/nrt2pvg/rma/1125_2023/MovieRenders_butterfly_1'
# root = jhelp(root)
myroots = []
if len(jhelp_folder(root))>0:
    myroots = jhelp_folder(root)
else:
    myroots = [root]

for myroot in myroots:
    files = sorted(jhelp_file(myroot))
    record = -1
    for file in files:
        nums = int(re.findall('\d+',file)[-1])
        if record <0 or nums==0:
            record = nums
            continue
        if nums - record !=1:
            name = os.path.basename(file).replace(str(nums),str(int(nums-1)))
            print(name)
        record = nums
