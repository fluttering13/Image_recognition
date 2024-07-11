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

def save_face_by_mtcnn(path,mtcnn):
    for i in range(1,41):
        for j in range(1,11):
            x= cv2.cvtColor(cv2.imread('att_faces/s{}/{}.pgm'.format(i,j)), cv2.COLOR_BGR2RGB)
            x_aligned, prob = mtcnn(x, return_prob=True, save_path=path+str(i)+'_'+str(j)+'.jpg')#prob is the confidence from face found by mtcnn
            #print(prob)
def show_the_images_and_embedding(path):
    aligned_list=[]
    names=[]
    for i in range(1,41):
        for j in range(1,11):
            #img=cv2.imread(path+str(i)+'_'+str(j)+'.jpg', cv2.COLOR_BGR2RGB) #show the img if necessary
            img=plt.imread(path+str(i)+'_'+str(j)+'.jpg')
            img=img.transpose(2,0,1) #change to cv2 format
            aligned_list.append(torch.from_numpy(img).to(torch.float32))
            names.append(i)

    aligned = torch.stack(aligned_list)
    embeddings = resnet(aligned).detach()
    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]

    print(pd.DataFrame(dists, columns=names, index=names)) #show the embedding


if __name__=='__main__':

    image_size=92

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu") #setting the cuda to inference with GPU

    resnet = InceptionResnetV1(pretrained='vggface2').eval() #choose the backbone to inference the facce

    mtcnn = MTCNN(image_size=image_size) #use the mt_cnn for the per-process and reshape to [92,92]

    path='./CPGAN_example/pre-process/pic'

    save_face_by_mtcnn(path,mtcnn) # save the result from mtcnn and whether the file existed

    #show_the_images_and_embedding(path) #show the result if necessary

