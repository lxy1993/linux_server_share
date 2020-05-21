# import tensorflow as tf
# import numpy as np
# import cv2
# #获取图像的基本
# img = cv2.imread('../out/2.jpg', 1)
# imgInfo = img.shape
# height = imgInfo[0]
# width =imgInfo[1]
# dstHeight = int(244)
# dstWidth = int(111)
# #创建空白模板，其中np.uint8代表图片的数据类型0-255
# dstImage = np.zeros((dstHeight, dstWidth, 3), np.uint8)
# #对新的图像坐标进行重新计算，对矩阵进行行列遍历
# for i in range(0, dstHeight):
#     for j in range(0, dstWidth):
#         iNew = int(i*(height*1.0 / dstHeight))
#         jNew = int(j*(width*1.0/dstWidth))
#         dstImage[i, j] = img[iNew, jNew]
# cv2.imshow('dst', dstImage)
# cv2.waitKey(0)
# import json
# candinate=[14,34,4.7,1]
# candinate1=[14,34,4.7,1]
# filename="skeleton.json"
# with open(filename,"w") as file_obj:
#     json.dump(candinate,file_obj)
#
# with open(filename,"r") as file_read:
#     sk=json.load(file_read)
# print(sk)
##########features scale############
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
