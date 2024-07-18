'''
Author: Qing Hong
Date: 2024-03-15 15:45:28
LastEditors: QingHong
LastEditTime: 2024-03-15 15:46:17
Description: file content
'''
a = None



def zxc(num):
    global a
    if a is None:
        a = 100
    a+=num
    print(a)

for i in range(100):
    zxc(i)