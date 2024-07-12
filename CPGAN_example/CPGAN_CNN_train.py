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

from CPGAN_CNN_net import Encoder, Classifier, init_weights
from CPGAN_utilty import compute_one_zero_acc, training_curry
import json


def read_data_before_training(training_dict):# load data in grayscale

    raw_train_data=[]
    train_true=[]
    raw_test_data=[]
    test_true=[]
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
            else:
                raw_test_data.append(data)
                test_true.append(i-1)

    raw_data=torch.from_numpy(np.array(raw_train_data)).float().reshape([200,1,92,92]).cuda()
    raw_data_train=torch.from_numpy(np.array(raw_train_data)).float().reshape([200,1,92,92]).cuda()
    labels=torch.from_numpy(np.array(train_true)).long().cuda()

    test_data=torch.from_numpy(np.array(raw_test_data))
    test_data=test_data.float().reshape([200,1,92,92]).cuda()

    training_dict['raw_train_data']=raw_train_data
    training_dict['raw_test_data']=raw_test_data
    training_dict['train_true']=train_true
    training_dict['test_true']=test_true
    training_dict['test_data']=test_data
    training_dict['raw_data']=raw_data
    training_dict['raw_data_train']=raw_data_train
    training_dict['labels']=labels
    
    return training_dict


def define_the_objs_in_training(training_dict):
    ###define the net
    encoder = Encoder().cuda()
    classifier=Classifier().cuda()

    ### define the loss function and optimizer
    ### 0:Classier, 1:constructor
    loss_CE_obj = nn.CrossEntropyLoss().cuda()
    loss_MSE_obj= nn.MSELoss().cuda()


    optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.001,  weight_decay=1e-6)
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=0.001,  weight_decay=1e-6)

    training_dict['encoder']=encoder
    training_dict['classifier']=classifier
    training_dict['loss_CE_obj']=loss_CE_obj
    training_dict['loss_MSE_obj']=loss_MSE_obj
    training_dict['optimizer_classifier']=optimizer_classifier
    training_dict['optimizer_encoder']=optimizer_encoder
    return training_dict

def CPGAN_processing(index,training_dict):

    optimizer_classifier=training_dict['optimizer_classifier']
    optimizer_encoder=training_dict['optimizer_encoder']
    classifier=training_dict['classifier']
    loss_CE_obj=training_dict['loss_CE_obj']
    loss_MSE_obj=training_dict['loss_MSE_obj']

    
    labels=training_dict['labels']

    trade_off_const=training_dict['trade_off_const']
    ###0:Classier:return the outcome of prediction
    if index=='classifier':# the classifer will be returned outcome of prediction
        data=training_dict['data_detach']
        loss_list=[]
        dif_loss=1
        loss_value=1
        count_dif_reverse=0
        count=0
        while True:
            optimizer_classifier.zero_grad()
            outputs=classifier(data)
            loss_classifer = loss_CE_obj(outputs, labels)
            loss_classifer.backward() 
            optimizer_classifier.step()
            loss_value=loss_classifer.item()
            #print('count',count,'loss',loss_value)
            if count==0:
                pass
            else:
                dif_loss=loss_list[-1]-loss_value
            if dif_loss<=1e-6:
               count_dif_reverse=count_dif_reverse+1
            if count_dif_reverse>=1000:
                break 
            
            loss_list.append(loss_value)
            count=count+1
            if count>=20000:
                break
        return  loss_classifer, outputs

    elif index=='reconstructor':
    ###1:reconstruction: reconstructor is designed to reconstruct the original input data with the embeding vector 
        #criterion1 = nn.MSELoss()
        data=training_dict['data_detach']
        raw_data=training_dict['raw_data']
        ridge_cof=0.001
        encoder_dimension=data.shape[1]

        raw_data=raw_data.reshape([200,-1])
        img_len=raw_data.shape[1]
        matrix_sum1=torch.zeros([encoder_dimension,encoder_dimension]).cuda()
        matrix_sum2=torch.zeros(encoder_dimension,img_len).cuda()
        for i in range(len(data)):
            vec_z=data[i,:].reshape([-1,1])
            vec_x=raw_data[i,:].reshape([-1,1])
            matrix_sum1=matrix_sum1+vec_z*vec_z.t()
            matrix_sum2=matrix_sum2+vec_z*vec_x.t()
        W_LRR=torch.matmul(torch.linalg.inv(matrix_sum1/len(data)+ridge_cof*torch.eye(encoder_dimension).cuda()),matrix_sum2/len(data))

        x_reconstructed=torch.matmul(W_LRR.t(),data.t())
        loss_reconstructor=loss_MSE_obj(x_reconstructed, raw_data.t())
        return loss_reconstructor, x_reconstructed
    
    elif index=='encoder': 
    ###2:encoder: finally, the encoder sum up the loss from classifer and reconstructor. Remember the copy is necessary then the back propagation is doable.
        data=training_dict['data_encoder']
        optimizer_classifier.zero_grad()
        outputs=classifier(data)
        loss_classifer = loss_CE_obj(outputs, labels)
        loss_reconstructor, x_reconstructed=CPGAN_processing('reconstructor',training_dict) #make copy of reconstructor to avoid double back propagation problems
        gan_loss=trade_off_const*loss_classifer-loss_reconstructor # combine two loss
        optimizer_encoder.zero_grad()
        gan_loss.backward() #back propagation
        optimizer_encoder.step()
        training_dict['data_encoder']=data
        return gan_loss
    else:
        return None
        

def CPGAN_training_epoches(training_dict):

    encoder=training_dict['encoder']
    classifier=training_dict['classifier']
    raw_data_train=training_dict['raw_data_train']
    labels=training_dict['labels']
    encoder_dimension=training_dict['encoder_dimension']
    training_process_times=training_dict['training_process_times']
    training_epoches=training_dict['training_epoches']
    test_data=training_dict['test_data']

    #initialize the list to save the checkpoints
    loss_reconstructor_list=[]
    err_list=[]
    acc_list=[]
    count=0
    for _ in range(training_process_times):
        init_weights(encoder)#start from different init params
        for _ in range(training_epoches):
            data_detach = encoder(raw_data_train).detach() # inputs of classifer, reconstructor
            data_encoder= encoder(raw_data_train) # inputs of decoder which have gradients     
            training_dict['data_detach']=data_detach
            training_dict['data_encoder']=data_encoder

            loss_classifer, outputs_classifer=CPGAN_processing('classifier',training_dict)
  

            embeding_from_test_data=encoder(test_data)
            outputs_test=classifier(embeding_from_test_data)

            label_trained=torch.argmax(outputs_classifer,1)
            label_test=torch.argmax(outputs_test,1)

            err=compute_one_zero_acc(labels,label_trained)
            acc=compute_one_zero_acc(labels,label_test)


            loss_reconstructor, x_reconstructed=CPGAN_processing('reconstructor',training_dict)

            gan_loss=CPGAN_processing('encoder',training_dict)
            loss_value=loss_reconstructor.item()

            print(err,acc,loss_value)
            err_list.append(err)
            acc_list.append(acc)
            loss_reconstructor_list.append(loss_value)      
            dic={'err_list':err_list,'acc_list':acc_list,'loss1_list':loss_reconstructor_list}
            fp=open('./check_points/CPGAN_CNN_d_'+str(encoder_dimension)+'.pkl', 'wb')
            pickle.dump(dic, fp)
            if count%10==0:
                fp=open('./check_points/CPGAN_CNN_d_'+str(encoder_dimension)+'backup'+'.pkl', 'wb')
                pickle.dump(dic, fp)
            count=count+1
    return training_dict




if __name__=='__main__':

        ###setting the training parameters from config.json
        training_dict={}
        config = json.load(open('config.json'))

        training_dict['encoder_dimension']=config["network_setting"]["encoder_dimension"]

        training_dict['trade_off_const']=config["training_setting"]["trade_off_const"]
        training_dict['training_process_times']=config["training_setting"]["training_process_times"]
        training_dict['training_epoches']=config["training_setting"]["training_epoches"]

        ###main functions
        training_processing=training_curry(
                    read_data_before_training,
                    define_the_objs_in_training,
                    CPGAN_training_epoches
                    )

        training_processing(training_dict)
