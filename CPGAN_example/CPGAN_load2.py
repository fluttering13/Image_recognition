import numpy as np
import pickle
import matplotlib.pyplot as plt

encoder_dimension=128
fp=open('./CPGAN_example/CPGAN_CNN_d_'+str(encoder_dimension)+'.pkl', 'rb')
data_128=pickle.load(fp)
encoder_dimension=256
fp=open('./CPGAN_example/CPGAN_CNN_d_'+str(encoder_dimension)+'.pkl', 'rb')
data_256=pickle.load(fp)


#print(data_128)
acc_list_128=data_128['acc_list']
loss1_list_128=data_128['loss1_list']
acc_list_256=data_256['acc_list']
loss1_list_256=data_256['loss1_list']

test_loss_list=loss1_list_128[600:700]
test_acc_list=acc_list_128[600:700]
trial=np.arange(100)

def privacy_acc_plot():
    plt.title('Privacy-acc',fontsize=20)
    plt.xlabel('Acc',fontsize=20)
    plt.ylabel('Privacy (l2 loss)',fontsize=20)
    plt.scatter(acc_list_256,loss1_list_256,c='g',label='d=256')
    plt.scatter(acc_list_128,loss1_list_128,c='b',label='d=128')
    plt.legend(
        loc='best',
        fontsize=12,
        shadow=True,
        facecolor='#ccc',
        edgecolor='#000',
        title='Encoded dimension',
        title_fontsize=12)
    plt.show()


def train_loss_polt():
    plt.title('Training_loss',fontsize=20)
    plt.xlabel('trail',fontsize=20)
    plt.ylabel('Privacy (l2 loss)',fontsize=20)
    plt.scatter(trial,test_loss_list,c='g',label='d=256')
    plt.show()

def train_acc_polt():
    plt.title('Training_acc',fontsize=20)
    plt.xlabel('trail',fontsize=20)
    plt.ylabel('Acc',fontsize=20)
    plt.scatter(trial,test_acc_list,c='g',label='d=256')
    plt.show()

train_acc_polt()
