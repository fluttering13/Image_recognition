import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
# from sklearn.svm import SVC
import numpy as np
import torch.optim as optim
import pickle
import cv2
# import scipy
# import optuna
# i=1
# j=5
# pic=plt.imread('./att_faces/s{}/{}.pgm'.format(i,j))
# print(pic.shape)
# plt.imshow(pic)  
# plt.show()

Encoder_dimension=128
trade_off_const=1

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5,kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=2000, out_features=Encoder_dimension)
        # self.fc = nn.Linear(1024, 1000)
        self.Dropout    = nn.Dropout(1 - 0.5)        
        # self.Bottleneck = nn.Linear(1024, 128,bias=False)
        self.last_bn = nn.BatchNorm1d(Encoder_dimension, eps=0.001, momentum=0.1, affine=True)
        #self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)       
        ###after the net drop and normalize
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
       # x = self.Dropout(x)
        #x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        #x = F.normalize(before_normalize, p=2, dim=1)

        return before_normalize


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    #print('initialize network with %s type' % init_type)
    net.apply(init_func)



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # self.conv1 = nn.Conv2d(1,6,5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(Encoder_dimension, 40)    
    def forward(self, x):
        x = self.fc(x)
        #x= F.softmax(x)
        return x




##RAWDATA
raw_train_data=[]
raw_test_data=[]
#ENCONDER
data=[]
train_true=[]
#test data
test_data=[]
test_true=[]
#container of storing training information
loss1_list=[]
loss2_list=[]
GAN_loss_list=[]
acc_list=[]

# print(np.eye(20)[0])
##41,11
for i in range(1,41):
    for j in range(1,11):
        #data=plt.imread('att_faces/s{}/{}.pgm'.format(i,j))
        #data='./CPGAN_example/pre-process/pic'+str(i)+'_'+str(j)+'.jpg'
        data=cv2.imread('./pre-process/pic'+str(i)+'_'+str(j)+'.jpg',cv2.IMREAD_GRAYSCALE)
        data=data.reshape([1,92,92])/255
        #print(data)

        if j <= 5:
            raw_train_data.append(data)
            train_true.append(i-1)
            #train_true_one_hot.append(np.eye(40)[i-1])
        else:
            raw_test_data.append(data)
            test_true.append(i-1)
            #test_true_one_hot.append(np.eye(40)[i-1])




###Classifier:SVM
# SVM = SVC(kernel='linear', decision_function_shape="ovo")
# SVM.fit(train_data, train_true)
# train_score = SVM.score(train_data, train_true)
# test_score = SVM.score(test_data, test_true)


def compare_labels(labels,test_label):
    correct=0
    for i in range(len(labels)):
        if test_label[i]==labels[i]:
            correct=correct+1
    acc=correct/len(labels)
    return acc

def oracle(index,data,labels,*raw_data):
    ###0:Classier:return the outcome of prediction
    if index==0:
        #criterion0 = nn.CrossEntropyLoss()
        #optimizer0 = optim.Adam(classifier.parameters(), lr=0.001,  weight_decay=1e-6)

        loss_list=[]
        dif_loss=1
        loss_value=1
        count_dif_reverse=0
        count=0
        while True:
            optimizer0.zero_grad()
            outputs=classifier(data)
            loss0 = criterion0(outputs, labels)
            loss0.backward() ###retain_graph=True
            optimizer0.step()
            loss_value=loss0.item()
            #print('count',count,'loss',loss_value)
            if count==0:
                pass
            else:
                dif_loss=loss_list[-1]-loss_value
            if dif_loss<=1e-6:
               count_dif_reverse=count_dif_reverse+1
            if count_dif_reverse>=1000:
                break 
            #print(count,dif_loss)
            loss_list.append(loss_value)
            count=count+1
            if count>=20000:
                break
        breakpoint()
        return  loss0, outputs

    elif index==1:
    ###1:reconstruction: return prediction and loss
        #criterion1 = nn.MSELoss()
        ridge_cof=0.001
        Encoder_dimension=data.shape[1]
        raw_data=raw_data[0]
        raw_data=raw_data.reshape([200,-1])
        img_len=raw_data.shape[1]
        matrix_sum1=torch.zeros([Encoder_dimension,Encoder_dimension]).cuda()
        matrix_sum2=torch.zeros(Encoder_dimension,img_len).cuda()
        for i in range(len(data)):
            vec_z=data[i,:].reshape([-1,1])
            vec_x=raw_data[i,:].reshape([-1,1])
            matrix_sum1=matrix_sum1+vec_z*vec_z.t()
            matrix_sum2=matrix_sum2+vec_z*vec_x.t()
        W_LRR=torch.matmul(torch.linalg.inv(matrix_sum1/len(data)+ridge_cof*torch.eye(Encoder_dimension).cuda()),matrix_sum2/len(data))

        x_reconstructed=torch.matmul(W_LRR.t(),data.t())
        loss1=criterion1(x_reconstructed, raw_data.t())
        return loss1, x_reconstructed
    elif index==2:

        optimizer0.zero_grad()
        outputs=classifier(data)
        loss0 = criterion0(outputs, labels)
        loss1, x_reconstructed=oracle(1,data,labels,*raw_data)
        gen_loss=trade_off_const*loss0-loss1
        optimizer2.zero_grad()
        gen_loss.backward()
        optimizer2.step()
        return gen_loss
    else:
        return None
        
    

###define the net

encoder = Encoder().cuda()

classifier=Classifier().cuda()


### define the loss function and optimizer
### 0:Classier, 1:constructor
criterion0 = nn.CrossEntropyLoss().cuda()
criterion1= nn.MSELoss().cuda()


optimizer0 = optim.Adam(classifier.parameters(), lr=0.001,  weight_decay=1e-6)
optimizer2 = optim.Adam(encoder.parameters(), lr=0.001,  weight_decay=1e-6)

raw_data=torch.from_numpy(np.array(raw_train_data)).float().reshape([200,1,92,92]).cuda()
raw_data_train=torch.from_numpy(np.array(raw_train_data)).float().reshape([200,1,92,92]).cuda()
labels=torch.from_numpy(np.array(train_true)).long().cuda()

test_data=torch.from_numpy(np.array(raw_test_data))
test_data=test_data.float().reshape([200,1,92,92]).cuda()
# weights_init(encoder)
##############################################
####training begin
err_list=[]
acc_list=[]
loss1_list=[]
count=0
for j in range(100):
    weights_init(encoder)
    for i in range(100):


        data = encoder(raw_data_train).detach()
        data2= encoder(raw_data_train)       



        loss0, outputs0=oracle(0,data,labels)


        code_test_data=encoder(test_data)


        outputs_test=classifier(code_test_data)
        label_trained=torch.argmax(outputs0,1)
        label_test=torch.argmax(outputs_test,1)

        err=compare_labels(labels,label_trained)
        acc=compare_labels(labels,label_test)




        loss1, x_reconstructed=oracle(1,data,labels,raw_data)

        gan_loss=oracle(2,data2,labels,raw_data)
        loss_value=loss1.item()

        print(err,acc,loss_value)
        err_list.append(err)
        acc_list.append(acc)
        loss1_list.append(loss_value)      
        dic={'err_list':err_list,'acc_list':acc_list,'loss1_list':loss1_list}
        fp=open('./CPGAN_CNN_d_'+str(Encoder_dimension)+'.pkl', 'wb')
        #pickle.dump(dic, fp)
        if count%10==0:
            fp=open('./CPGAN_CNN_d_'+str(Encoder_dimension)+'backup'+'.pkl', 'wb')
            #pickle.dump(dic, fp)
        count=count+1






# for parameters in classifier.parameters():
#     print(parameters)