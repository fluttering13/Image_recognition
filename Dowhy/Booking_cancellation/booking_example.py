import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dowhy
from IPython.display import Image, display

dataset = pd.read_csv('https://raw.githubusercontent.com/Sid-darthvader/DoWhy-The-Causal-Story-Behind-Hotel-Booking-Cancellations/master/hotel_bookings.csv')
print(dataset.shape)
print(dataset.columns)


'''
feature engineering
'''
# Total stay in nights
dataset['total_stay'] = dataset['stays_in_week_nights']+dataset['stays_in_weekend_nights']
# Total number of guests
dataset['guests'] = dataset['adults']+dataset['children'] +dataset['babies']
# Creating the different_room_assigned feature
dataset['different_room_assigned']=0
slice_indices =dataset['reserved_room_type']!=dataset['assigned_room_type']
dataset.loc[slice_indices,'different_room_assigned']=1
# Deleting older features
dataset = dataset.drop(['stays_in_week_nights','stays_in_weekend_nights','adults','children','babies'
                        ,'reserved_room_type','assigned_room_type'],axis=1)

dataset.isnull().sum() # Country,Agent,Company contain 488,16340,112593 missing entries 
dataset = dataset.drop(['agent','company'],axis=1)
# Replacing missing countries with most freqently occuring countries
dataset['country']= dataset['country'].fillna(dataset['country'].mode()[0])
dataset = dataset.drop(['reservation_status','reservation_status_date','arrival_date_day_of_month'],axis=1)
dataset = dataset.drop(['arrival_date_year'],axis=1)
dataset = dataset.drop(['distribution_channel'], axis=1)
dataset['different_room_assigned']= dataset['different_room_assigned'].replace(1,True)
dataset['different_room_assigned']= dataset['different_room_assigned'].replace(0,False)
dataset['is_canceled']= dataset['is_canceled'].replace(1,True)
dataset['is_canceled']= dataset['is_canceled'].replace(0,False)
dataset.dropna(inplace=True)
# dataset = dataset[dataset.deposit_type=="No Deposit"]
# print(dataset.groupby(['deposit_type','is_canceled']).count())
dataset_copy = dataset.copy(deep=True)

'''
choose 1000 observations at random
'is_cancelled' & 'different_room_assigned' 

'''

# counts_sum=0
# for i in range(1,10000):
#         counts_i = 0
#         rdf = dataset.sample(1000)
#         counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
#         counts_sum+= counts_i
# print(counts_sum/10000)

# counts_sum=0
# for i in range(1,10000):
#         counts_i = 0
#         rdf = dataset[dataset["booking_changes"]>0].sample(1000)
#         counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
#         counts_sum+= counts_i
# print(counts_sum/10000)

causal_graph = """digraph {
different_room_assigned[label="Different Room Assigned"];
is_canceled[label="Booking Cancelled"];
booking_changes[label="Booking Changes"];
previous_bookings_not_canceled[label="Previous Booking Retentions"];
days_in_waiting_list[label="Days in Waitlist"];
lead_time[label="Lead Time"];
market_segment[label="Market Segment"];
country[label="Country"];
U[label="Unobserved Confounders",observed="no"];
is_repeated_guest;
total_stay;
guests;
meal;
hotel;
U->{different_room_assigned,required_car_parking_spaces,guests,total_stay,total_of_special_requests};
market_segment -> lead_time;
lead_time->is_canceled; country -> lead_time;
different_room_assigned -> is_canceled;
country->meal;
lead_time -> days_in_waiting_list;
days_in_waiting_list ->{is_canceled,different_room_assigned};
previous_bookings_not_canceled -> is_canceled;
previous_bookings_not_canceled -> is_repeated_guest;
is_repeated_guest -> {different_room_assigned,is_canceled};
total_stay -> is_canceled;
guests -> is_canceled;
booking_changes -> different_room_assigned; booking_changes -> is_canceled; 
hotel -> {different_room_assigned,is_canceled};
required_car_parking_spaces -> is_canceled;
total_of_special_requests -> {booking_changes,is_canceled};
country->{hotel, required_car_parking_spaces,total_of_special_requests};
market_segment->{hotel, required_car_parking_spaces,total_of_special_requests};
}"""
model= dowhy.CausalModel(
        data = dataset,
        graph=causal_graph.replace("\n", " "),
        treatment="different_room_assigned",
        outcome='is_canceled')

# model.view_model()
# display(Image(filename="causal_model.png",width=600))

'''
remove the unobserved confounder node
check which variables can' be identified 
'''
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor.propensity_score_weighting",target_units="ate")
# ATE = Average Treatment Effect
# ATT = Average Treatment Effect on Treated (i.e. those who were assigned a different room)
# ATC = Average Treatment Effect on Control (i.e. those who were not assigned a different room)
print(estimate)



'''
Random Common Cause:
- Adds randomly drawn covariates to data and re-runs the analysis to see if the causal estimate changes or not. 
If our assumption was originally correct then the causal estimate shouldn't change by much.
'''
refute1_results=model.refute_estimate(identified_estimand, estimate,
        method_name="random_common_cause")
print(refute1_results)

'''
Placebo Treatment Refuter:- Randomly assigns any covariate as a treatment and re-runs the analysis. 
If our assumptions were correct then this newly found out estimate should go to 0.
'''
refute2_results=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter")
print(refute2_results)
'''
Data Subset Refuter:- Creates subsets of the data(similar to cross-validation) 
and checks whether the causal estimates vary across subsets. 
If our assumptions were correct there shouldn't be much variation.
'''
refute3_results=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter")
print(refute3_results)

