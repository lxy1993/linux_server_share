import  matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D


skeleton_all =np.load("all_tracklet.npy")
print(skeleton_all.shape)
# print(skeleton_all[0])
######x,y coordinate#######
# x_coordinate =skeleton_all[0,:,0]
# y_coordinate =skeleton_all[0,:,1]
# print(x_coordinate.shape,y_coordinate.shape)

# plt,axis=plt.subplot(1,2,figsize=(6,6))
# axis[0,0].scatter(x_coordinate,y_coordinate)
# axis[0,1].hist2d(x_coordinate,y_coordinate)
fig =plt.figure()
ax =Axes3D(fig)
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],\
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],\
           [1, 16], [16, 18]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],\
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],\
          [170, 0, 255], [255, 0, 255]]


for i in [0]:
    for j in range(len(limbSeq)):
        index =[np.array(limbSeq[j]) - 1]
        kp1=skeleton_all[i,index[0][0],:]
        kp2=skeleton_all[i,index[0][1],:]
        x=np.array([kp1[0],kp2[0]])
        y = np.array([kp1[1], kp2[1]])
        ax.plot(x,y,zs=0,zdir="z")

    # x_coordinate = skeleton_all[i, :, 0]
    # y_coordinate = i
    # z_coordinate =skeleton_all[i, :, 1]
#
#     ax.scatter(x_coordinate,y_coordinate,zs=z_coordinate)
#
# ax.legend()
# ax.set_xlim(-10,10)
# ax.set_zlim(-10,10)
# ax.set_ylim(-5,30)
#
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
#
ax.view_init(elev=-158,azim=-104)
plt.show()

