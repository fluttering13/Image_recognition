import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
#from sklearn.svm import SVC
import numpy as np
import torch.optim as optim
import pickle
# import scipy
# import optuna
# i=1
# j=5
# pic=plt.imread('./att_faces/s{}/{}.pgm'.format(i,j))
# print(pic.shape)
# plt.imshow(pic)  
# plt.show()

Encoder_dimension=16
trade_off_const=1
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(112*92, Encoder_dimension)
    def forward(self, x):
        #x = torch.flatten(x) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

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
        data=plt.imread('att_faces/s{}/{}.pgm'.format(i,j))
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
            
            #print('count',count,'loss',loss_value)
            if count==0:
                pass
            else:
                dif_loss=loss_list[-1]-loss0.item()
            if dif_loss<=1e-7:
               count_dif_reverse=count_dif_reverse+1
            if count_dif_reverse>=2000:
                break 
            #print(count,dif_loss)
            loss_list.append(loss0.item())
            count=count+1
            if count>=20000:
                break
            
        return  loss0, outputs

    elif index==1:
    ###1:reconstruction: return prediction and loss
        #criterion1 = nn.MSELoss()
        ridge_cof=0.001
        Encoder_dimension=data.shape[1]
        raw_data=raw_data[0]
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
        
    
use_gpu = torch.cuda.is_available()
print(use_gpu)
###define the net

encoder = Encoder().cuda()
# for m in encoder.modules():
#     if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
#         torch.nn.init.xavier_uniform_(m.weight)
classifier=Classifier().cuda()
# for m in classifier.modules():
#     if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
#         torch.nn.init.xavier_uniform_(m.weight)

### define the loss function and optimizer
### 0:Classier, 1:constructor
criterion0 = nn.CrossEntropyLoss().cuda()
criterion1= nn.MSELoss().cuda()


optimizer0 = optim.Adam(classifier.parameters(), lr=0.001,  weight_decay=1e-6)
optimizer2 = optim.Adam(encoder.parameters(), lr=0.001,  weight_decay=1e-6)

raw_data=torch.from_numpy(np.array(raw_train_data)).float().reshape([200,-1]).cuda()


##############################################
####training begin
for Encoder_dimension in [8192,16384,32768,65536,131072,262144]:
    err_list=[]
    acc_list=[]
    loss1_list=[]
    for i in range(100):
        data = encoder(raw_data).detach()
        data2= encoder(raw_data)       

        labels=torch.from_numpy(np.array(train_true)).long().cuda()





        loss0, outputs0=oracle(0,data,labels)

        test_data=torch.from_numpy(np.array(raw_test_data))
        test_data=test_data.float().reshape([200,-1]).cuda()
        test_data=encoder(test_data)

        outputs_test=classifier(test_data)

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
        fp=open('./CPGAN_example/CPGAN_prototype_d_'+str(Encoder_dimension)+'.pkl', 'wb')
        pickle.dump(dic, fp)




# for parameters in classifier.parameters():
#     print(parameters)