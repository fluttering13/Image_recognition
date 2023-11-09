import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas import DataFrame as Df
import pickle


'''
Content
Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

The data set includes information about:

Customers who left within the last month – the column is called Churn
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers – gender, age range, and if they have partners and dependents
Inspiration
To explore this type of models and learn more about the subject.
'''

path='./Customer-Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
f_path='./Customer-Churn/model_save/'
data_pd=pd.read_csv(path)

### list the columns
columns=data_pd.columns

def basic_information():
    #Find the missing part of the data
    #缺失值填補
    ##數據完整其實不需要填補
    # for name in columns:
    #     cond1=data_pd.isnull()[name]==True
    #     print(data_pd[cond1])
    # ##或者是這樣
    # print(data_pd.isnull().sum())

    ##列表uniuqe的元素
    print(data_pd.nunique())
    for name in data_pd.columns:
        print(pd.unique(data_pd[name]))

    ## categorical data
    for col in columns:
        print(data_pd[col].value_counts())


def create_intervention_AB(data_number,data_dead_number):

    data_number=data_number.assign(total=data_number.sum(axis=1))
    data_dead_number=data_dead_number.assign(total=data_dead_number.sum(axis=1))

    indexes=data_number.index
    data_rates=data_dead_number/data_number

    data_rates=data_rates.rename(columns={'total':'correlation'})


    new_list=[]
    c_numbers=data_number.sum()
    for i in range(data_rates.shape[0]):
        tmp_sum=0
        for j in range(data_rates.shape[1]-1):
            E_y_given_t_c=data_rates.iloc[i,j]
            p_c=c_numbers.iloc[j]/c_numbers.iloc[-1]
            tmp_sum=tmp_sum+E_y_given_t_c*p_c
        new_list.append(tmp_sum)
    new_list=pd.DataFrame({'intervention':new_list},index=indexes)
    data_rates=pd.concat([data_rates,new_list],axis=1)

    return data_rates
# basic_information()

###data type transfom and numericalize
data_pd_numberical=data_pd.replace({'Yes':1,'No':0,'No internet service':2,'No phone service':2,'DSL':1,'Fiber optic':2,'Month-to-month':0,'One year':1,'Two year':2})
data_pd_numberical=data_pd_numberical.replace({'Electronic check':0,'Mailed check':1,'Bank transfer (automatic)':2,'Credit card (automatic)':3})
# ###numberical feature
# data_pd_numberical=data_pd[['tenure','MonthlyCharges','TotalCharges']]


MC_ave=data_pd_numberical['MonthlyCharges'].mean()


# data_pd_numberical['tenure']=pd.to_numeric(data_pd_numberical['tenure'])

def tenure_causal_test():
    ave=data_pd_numberical['tenure'].mean()

    cond_list=[]  
    cond_list.append((data_pd_numberical['tenure']>=ave) & (data_pd_numberical['MonthlyCharges']>=MC_ave))
    cond_list.append((data_pd_numberical['tenure']>=ave) & (data_pd_numberical['MonthlyCharges']<MC_ave))
    cond_list.append((data_pd_numberical['tenure']<ave) & (data_pd_numberical['MonthlyCharges']>=MC_ave))
    cond_list.append((data_pd_numberical['tenure']<ave) & (data_pd_numberical['MonthlyCharges']<MC_ave))

    cond_list2=[]  
    cond_list2.append((data_pd_numberical['tenure']>=ave) & (data_pd_numberical['MonthlyCharges']>=MC_ave) & (data_pd_numberical['Churn']==1))
    cond_list2.append((data_pd_numberical['tenure']>=ave) & (data_pd_numberical['MonthlyCharges']<MC_ave) & (data_pd_numberical['Churn']==1))
    cond_list2.append((data_pd_numberical['tenure']<ave) & (data_pd_numberical['MonthlyCharges']>=MC_ave) & (data_pd_numberical['Churn']==1))
    cond_list2.append((data_pd_numberical['tenure']<ave) & (data_pd_numberical['MonthlyCharges']<MC_ave) & (data_pd_numberical['Churn']==1))



    highMC_list=[]
    lowMC_list=[]
    highMC_list.append(data_pd_numberical[cond_list[0]]['customerID'].count())
    highMC_list.append(data_pd_numberical[cond_list[2]]['customerID'].count())
    lowMC_list.append(data_pd_numberical[cond_list[1]]['customerID'].count())
    lowMC_list.append(data_pd_numberical[cond_list[3]]['customerID'].count())
    highMC_list2=[]
    lowMC_list2=[]
    highMC_list2.append(data_pd_numberical[cond_list2[0]]['customerID'].count())
    highMC_list2.append(data_pd_numberical[cond_list2[2]]['customerID'].count())
    lowMC_list2.append(data_pd_numberical[cond_list2[1]]['customerID'].count())
    lowMC_list2.append(data_pd_numberical[cond_list2[3]]['customerID'].count())


    total_df=Df({'h_MC':highMC_list,'l_MC':lowMC_list})
    df=Df({'h_MC':highMC_list2,'l_MC':lowMC_list2})

    data_rates=create_intervention_AB(total_df,df)
    print(data_rates)



def partner_causal_test():
    cond_list=[]  
    cond_list.append((data_pd_numberical['Partner']==1) & (data_pd_numberical['MonthlyCharges']>=MC_ave))
    cond_list.append((data_pd_numberical['Partner']==1) & (data_pd_numberical['MonthlyCharges']<MC_ave))
    cond_list.append((data_pd_numberical['Partner']==0) & (data_pd_numberical['MonthlyCharges']>=MC_ave))
    cond_list.append((data_pd_numberical['Partner']==0) & (data_pd_numberical['MonthlyCharges']<MC_ave))

    cond_list2=[]  
    cond_list2.append((data_pd_numberical['Partner']==1) & (data_pd_numberical['MonthlyCharges']>=MC_ave)& (data_pd_numberical['Churn']==1))
    cond_list2.append((data_pd_numberical['Partner']==1) & (data_pd_numberical['MonthlyCharges']<MC_ave)& (data_pd_numberical['Churn']==1))
    cond_list2.append((data_pd_numberical['Partner']==0) & (data_pd_numberical['MonthlyCharges']>=MC_ave)& (data_pd_numberical['Churn']==1))
    cond_list2.append((data_pd_numberical['Partner']==0) & (data_pd_numberical['MonthlyCharges']<MC_ave)& (data_pd_numberical['Churn']==1))

    highMC_list=[]
    lowMC_list=[]
    highMC_list.append(data_pd_numberical[cond_list[0]]['customerID'].count())
    highMC_list.append(data_pd_numberical[cond_list[2]]['customerID'].count())
    lowMC_list.append(data_pd_numberical[cond_list[1]]['customerID'].count())
    lowMC_list.append(data_pd_numberical[cond_list[3]]['customerID'].count())
    highMC_list2=[]
    lowMC_list2=[]
    highMC_list2.append(data_pd_numberical[cond_list2[0]]['customerID'].count())
    highMC_list2.append(data_pd_numberical[cond_list2[2]]['customerID'].count())
    lowMC_list2.append(data_pd_numberical[cond_list2[1]]['customerID'].count())
    lowMC_list2.append(data_pd_numberical[cond_list2[3]]['customerID'].count())


    total_df=Df({'h_MC':highMC_list,'l_MC':lowMC_list})
    df=Df({'h_MC':highMC_list2,'l_MC':lowMC_list2})

    data_rates=create_intervention_AB(total_df,df)
    print(data_rates)


def payment_causal_test():
    cond_list=[]
    for i in range(4):  
        cond_list.append((data_pd_numberical['PaymentMethod']==i) & (data_pd_numberical['MonthlyCharges']>=MC_ave))
        cond_list.append((data_pd_numberical['PaymentMethod']==i) & (data_pd_numberical['MonthlyCharges']<MC_ave))
    cond_list2=[]
    for i in range(4):  
        cond_list2.append((data_pd_numberical['PaymentMethod']==i) & (data_pd_numberical['MonthlyCharges']>=MC_ave)& (data_pd_numberical['Churn']==1))
        cond_list2.append((data_pd_numberical['PaymentMethod']==i) & (data_pd_numberical['MonthlyCharges']<MC_ave)& (data_pd_numberical['Churn']==1))
    highMC_list=[]
    lowMC_list=[]
    highMC_list2=[]
    lowMC_list2=[]

    for i in range(0,8,2):
        highMC_list.append(data_pd_numberical[cond_list[i]]['customerID'].count())
        highMC_list2.append(data_pd_numberical[cond_list2[i]]['customerID'].count())
    for i in range(1,8,2):
        lowMC_list.append(data_pd_numberical[cond_list[i]]['customerID'].count())
        lowMC_list2.append(data_pd_numberical[cond_list2[i]]['customerID'].count())
    total_df=Df({'h_MC':highMC_list,'l_MC':lowMC_list})
    df=Df({'h_MC':highMC_list2,'l_MC':lowMC_list2})
    data_rates=create_intervention_AB(total_df,df)
    print(data_rates)

tenure_causal_test()
partner_causal_test()
payment_causal_test()

