import torch
from torch.nn import functional as F
import torch.nn as nn
from facenet_pytorch import MTCNN,InceptionResnetV1
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from sklearn.svm import SVC
import numpy as np
import torch.optim as optim
from PIL import Image
import cv2
import pandas as pd


###resnet feed with resnet
resnet = InceptionResnetV1(pretrained='vggface2').eval()
###########use the mt_cnn for the per-process and reshape to [92,92]
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

#####mtcnn feed with (X,Y,C)
mtcnn = MTCNN(image_size=92)
##############################################

###########find_the_face##############################
# path='./CPGAN_example/pre-process/pic'
# for i in range(1,41):
#     for j in range(1,11):
#         x= cv2.cvtColor(cv2.imread('att_faces/s{}/{}.pgm'.format(i,j)), cv2.COLOR_BGR2RGB)
#         x_aligned, prob = mtcnn(x, return_prob=True,save_path=path+str(i)+'_'+str(j)+'.jpg')
#######################################################
path='./CPGAN_example/pre-process/pic'

aligned_list=[]
names=[]
for i in range(1,41):
    for j in range(1,11):
        #img=cv2.imread(path+str(i)+'_'+str(j)+'.jpg', cv2.COLOR_BGR2RGB)
        img=plt.imread('./CPGAN_example/pre-process/pic'+str(i)+'_'+str(j)+'.jpg')
        #print(img.shape)
        img=img.transpose(2,0,1)
        aligned_list.append(torch.from_numpy(img).to(torch.float32))
        names.append(i)
print(img.shape)
aligned = torch.stack(aligned_list)
embeddings = resnet(aligned).detach()
print(embeddings)
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]

print(pd.DataFrame(dists, columns=names, index=names))



'''
Triple-loss:
Anchor: the vector from training
positive: belong the same label
negative: not the same label
L= max(d(a,p)-d(a,n),0)
'''
