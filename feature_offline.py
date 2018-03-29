import pandas as pd
import numpy as np
from datetime import date
pd.options.mode.chained_assignment = None  # default='warn'

"""
dataset split:
                      (date_received)                              
           dateset3: 20160701~20160731 (113640),features3 from 20160315~20160630  (off_test)
           dateset2: 20160515~20160615 (258446),features2 from 20160201~20160514  
           dateset1: 20160414~20160514 (138303),features1 from 20160101~20160413        



1.merchant related: 
      sales_use_coupon. total_coupon
      transfer_rate = sales_use_coupon/total_coupon.
      merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon 
      total_sales.  coupon_rate = sales_use_coupon/total_sales.  
       
2.coupon related: 
      discount_rate. discount_man. discount_jian. is_man_jian
      day_of_week,day_of_month. (date_received)
      
3.user related: 
      distance. 
      user_avg_distance, user_min_distance,user_max_distance. 
      buy_use_coupon. buy_total. coupon_received.
      buy_use_coupon/coupon_received. 
      avg_diff_date_datereceived. min_diff_date_datereceived. max_diff_date_datereceived.  
      count_merchant.  

4.user_merchant:
      times_user_buy_merchant_before.
     

5. other feature:
      this_month_user_receive_all_coupon_count
      this_month_user_receive_same_coupon_count
      this_month_user_receive_same_coupon_lastone
      this_month_user_receive_same_coupon_firstone
      this_day_user_receive_all_coupon_count
      this_day_user_receive_same_coupon_count
      day_gap_before, day_gap_after  (receive the same coupon)
"""


#1754884 record,1053282 with coupon_id,9738 coupon. date_received:20160101~20160615,date:20160101~20160630, 539438 users, 8415 merchants
off_train = pd.read_csv('data/ccf_offline_stage1_train.csv',header=0,dtype=str)
off_train.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']
#2050 coupon_id. date_received:20160701~20160731, 76309 users(76307 in trainset, 35965 in online_trainset), 1559 merchants(1558 in trainset)
off_test = pd.read_csv('data/ccf_offline_stage1_test_revised.csv',header=0,dtype=str)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']
#11429826 record(872357 with coupon_id),762858 user(267448 in off_train)
on_train = pd.read_csv('data/ccf_online_stage1_train.csv',header=0,dtype=str)
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']


dataset3 = off_test
feature3 = off_train[((off_train.date>='20160415')&(off_train.date<='20160630'))|((pd.isnull(off_train.date))&(off_train.date_received>='20160415')&(off_train.date_received<='20160630'))]
dataset2 = off_train[(off_train.date_received>='20160515')&(off_train.date_received<='20160615')]
feature2 = off_train[(off_train.date>='20160301')&(off_train.date<='20160514')|((pd.isnull(off_train.date))&(off_train.date_received>='20160301')&(off_train.date_received<='20160514'))]
dataset1 = off_train[(off_train.date_received>='20160414')&(off_train.date_received<='20160514')]
feature1 = off_train[(off_train.date>='20160201')&(off_train.date<='20160413')|((pd.isnull(off_train.date))&(off_train.date_received>='20160201')&(off_train.date_received<='20160413'))]
dataset0 = off_train[(off_train.date_received>='20160314')&(off_train.date_received<='20160414')]
feature0 = off_train[((off_train.date>='20160101')&(off_train.date<='20160313'))|((pd.isnull(off_train.date))&(off_train.date_received>='20160101')&(off_train.date_received<='20160313'))]

############# other feature ##################3
"""
5. other feature:
      this_month_user_receive_all_coupon_count
      this_month_user_receive_same_coupon_count
      this_month_user_receive_same_coupon_lastone
      this_month_user_receive_same_coupon_firstone
      this_day_user_receive_all_coupon_count
      this_day_user_receive_same_coupon_count
      day_gap_before, day_gap_after  (receive the same coupon)
"""

#for dataset3
t = dataset3[['user_id']]
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby('user_id').agg('sum').reset_index()

t1 = dataset3[['user_id','coupon_id']]
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()

t2 = dataset3[['user_id','coupon_id','date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':')))
t2 = t2[t2.receive_number>1]
t2['max_date_received'] = t2.date_received.apply(lambda s:max([int(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(lambda s:min([int(d) for d in s.split(':')]))
t2 = t2[['user_id','coupon_id','max_date_received','min_date_received']]

t3 = dataset3[['user_id','coupon_id','date_received']]
t3 = pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
t31 = t3.copy()
t3['date_received'] = t3.date_received.astype(float)
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received - t3.min_date_received
t3['date_received'] = t31['date_received']

def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        return -1 #those only receive once
        
t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
t3 = t3[['user_id','coupon_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]

t4 = dataset3[['user_id','date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id','date_received']).agg('sum').reset_index()

t5 = dataset3[['user_id','coupon_id','date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()

t6 = dataset3[['user_id','coupon_id','date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t6.rename(columns={'date_received':'dates'},inplace=True)

def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
        
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]),int(d[4:6]),int(d[6:8]))-date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
    

t7 = dataset3[['user_id','coupon_id','date_received']]
t7 = pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]

other_feature3 = pd.merge(t1,t,on='user_id')
other_feature3 = pd.merge(other_feature3,t3,on=['user_id','coupon_id'])
other_feature3 = pd.merge(other_feature3,t4,on=['user_id','date_received'])
other_feature3 = pd.merge(other_feature3,t5,on=['user_id','coupon_id','date_received'])
other_feature3 = pd.merge(other_feature3,t7,on=['user_id','coupon_id','date_received'])
print('other_feature3',other_feature3.shape)
other_feature3.to_csv('data/other_feature3.csv',index=None)



#for dataset2
t = dataset2[['user_id']]
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby('user_id').agg('sum').reset_index()

t1 = dataset2[['user_id','coupon_id']]
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()

t2 = dataset2[['user_id','coupon_id','date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':')))
t2 = t2[t2.receive_number>1]
t2['max_date_received'] = t2.date_received.apply(lambda s:max([int(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(lambda s:min([int(d) for d in s.split(':')]))
t2 = t2[['user_id','coupon_id','max_date_received','min_date_received']]

t3 = dataset2[['user_id','coupon_id','date_received']]
t3 = pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
t32 = t3.copy()
t3['date_received'] = t3.date_received.astype(float)
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received - t3.min_date_received
t3['date_received'] = t32['date_received']
def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        return -1 #those only receive once
        
t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
t3 = t3[['user_id','coupon_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]

t4 = dataset2[['user_id','date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id','date_received']).agg('sum').reset_index()

t5 = dataset2[['user_id','coupon_id','date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()

t6 = dataset2[['user_id','coupon_id','date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t6.rename(columns={'date_received':'dates'},inplace=True)

def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
        
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]),int(d[4:6]),int(d[6:8]))-date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
    

t7 = dataset2[['user_id','coupon_id','date_received']]
t7 = pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]

other_feature2 = pd.merge(t1,t,on='user_id')
other_feature2 = pd.merge(other_feature2,t3,on=['user_id','coupon_id'])
other_feature2 = pd.merge(other_feature2,t4,on=['user_id','date_received'])
other_feature2 = pd.merge(other_feature2,t5,on=['user_id','coupon_id','date_received'])
other_feature2 = pd.merge(other_feature2,t7,on=['user_id','coupon_id','date_received'])
print('other_feature2',other_feature2.shape)
other_feature2.to_csv('data/other_feature2.csv',index=None)


#for dataset1
t = dataset1[['user_id']]
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby('user_id').agg('sum').reset_index()

t1 = dataset1[['user_id','coupon_id']]
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()

t2 = dataset1[['user_id','coupon_id','date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':')))
t2 = t2[t2.receive_number>1]
t2['max_date_received'] = t2.date_received.apply(lambda s:max([int(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(lambda s:min([int(d) for d in s.split(':')]))
t2 = t2[['user_id','coupon_id','max_date_received','min_date_received']]

t3 = dataset1[['user_id','coupon_id','date_received']]
t3 = pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
t31 = t3.copy()
t3['date_received'] = t3.date_received.astype(float)
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received - t3.min_date_received
t3['date_received'] = t31['date_received']
def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        return -1 #those only receive once
        
t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
t3 = t3[['user_id','coupon_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]
t4 = dataset1[['user_id','date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id','date_received']).agg('sum').reset_index()

t5 = dataset1[['user_id','coupon_id','date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()

t6 = dataset1[['user_id','coupon_id','date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t6.rename(columns={'date_received':'dates'},inplace=True)

def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
        
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]),int(d[4:6]),int(d[6:8]))-date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
    

t7 = dataset1[['user_id','coupon_id','date_received']]
t7 = pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]

other_feature1 = pd.merge(t1,t,on='user_id')
other_feature1 = pd.merge(other_feature1,t3,on=['user_id','coupon_id'])
other_feature1 = pd.merge(other_feature1,t4,on=['user_id','date_received'])
other_feature1 = pd.merge(other_feature1,t5,on=['user_id','coupon_id','date_received'])
other_feature1 = pd.merge(other_feature1,t7,on=['user_id','coupon_id','date_received'])
print('other_feature1',other_feature1.shape)
other_feature1.to_csv('data/other_feature1.csv',index=None)

# for dataset0
t = dataset0[['user_id']]
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby('user_id').agg('sum').reset_index()

t1 = dataset0[['user_id', 'coupon_id']]
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

t2 = dataset0[['user_id', 'coupon_id', 'date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
t2 = t2[t2.receive_number > 1]
t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

t3 = dataset0[['user_id', 'coupon_id', 'date_received']]
t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
t31 = t3.copy()
t3['date_received'] = t3.date_received.astype(float)
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received - t3.min_date_received
t3['date_received'] = t31['date_received']


def is_firstlastone(x):
    if x == 0:
        return 1
    elif x > 0:
        return 0
    else:
        return -1  # those only receive once


t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone',
         'this_month_user_receive_same_coupon_firstone']]
t4 = dataset0[['user_id', 'date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

t5 = dataset0[['user_id', 'coupon_id', 'date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()

t6 = dataset0[['user_id', 'coupon_id', 'date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
t6.rename(columns={'date_received': 'dates'}, inplace=True)


def get_day_gap_before(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (
        date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8])) - date(int(d[0:4]), int(d[4:6]),
                                                                                               int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


def get_day_gap_after(s):
    date_received, dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]), int(d[4:6]), int(d[6:8])) - date(int(date_received[0:4]), int(date_received[4:6]),
                                                                       int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)


t7 = dataset0[['user_id', 'coupon_id', 'date_received']]
t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]

other_feature0 = pd.merge(t1, t, on='user_id')
other_feature0 = pd.merge(other_feature0, t3, on=['user_id', 'coupon_id'])
other_feature0 = pd.merge(other_feature0, t4, on=['user_id', 'date_received'])
other_feature0 = pd.merge(other_feature0, t5, on=['user_id', 'coupon_id', 'date_received'])
other_feature0 = pd.merge(other_feature0, t7, on=['user_id', 'coupon_id', 'date_received'])
print('other_feature0',other_feature0.shape)
other_feature0.to_csv('data/other_feature0.csv', index=None)

############# coupon related feature   #############
"""
2.coupon related: 
      discount_rate. discount_man. discount_jian. is_man_jian
      day_of_week,day_of_month. (date_received)
"""
def calc_discount_rate(s):
    s =str(s)
    s = s.split(':')
    if len(s)==1:
        return float(s[0])
    else:
        return 1.0-float(s[1])/float(s[0])

def get_discount_man(s):
    s =str(s)
    s = s.split(':')
    if len(s)==1:
        return 'null'
    else:
        return int(s[0])
        
def get_discount_jian(s):
    s =str(s)
    s = s.split(':')
    if len(s)==1:
        return 'null'
    else:
        return int(s[1])

def is_man_jian(s):
    s =str(s)
    s = s.split(':')
    if len(s)==1:
        return 0
    else:
        return 1

#dataset3
dataset3['day_of_week'] = dataset3.date_received.astype('str').apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
dataset3['day_of_month'] = dataset3.date_received.astype('str').apply(lambda x:int(x[6:8]))
dataset3['days_distance'] = dataset3.date_received.astype('str').apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,6,30)).days)
dataset3['discount_man'] = dataset3.discount_rate.apply(get_discount_man)
dataset3['discount_jian'] = dataset3.discount_rate.apply(get_discount_jian)
dataset3['is_man_jian'] = dataset3.discount_rate.apply(is_man_jian)
dataset3['discount_rate'] = dataset3.discount_rate.apply(calc_discount_rate)
d = dataset3[['coupon_id']]
d['coupon_count'] = 1
d = d.groupby('coupon_id').agg('sum').reset_index()
dataset3 = pd.merge(dataset3,d,on='coupon_id',how='left')
print('coupon_ralated3',dataset3.shape)
dataset3.to_csv('data/coupon3_feature.csv',index=None)
#dataset2
dataset2['day_of_week'] = dataset2.date_received.astype('str').apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
dataset2['day_of_month'] = dataset2.date_received.astype('str').apply(lambda x:int(x[6:8]))
dataset2['days_distance'] = dataset2.date_received.astype('str').apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,5,14)).days)
dataset2['discount_man'] = dataset2.discount_rate.apply(get_discount_man)
dataset2['discount_jian'] = dataset2.discount_rate.apply(get_discount_jian)
dataset2['is_man_jian'] = dataset2.discount_rate.apply(is_man_jian)
dataset2['discount_rate'] = dataset2.discount_rate.apply(calc_discount_rate)
d = dataset2[['coupon_id']]
d['coupon_count'] = 1
d = d.groupby('coupon_id').agg('sum').reset_index()
dataset2 = pd.merge(dataset2,d,on='coupon_id',how='left')
print('coupon_ralated2',dataset2.shape)
dataset2.to_csv('data/coupon2_feature.csv',index=None)
#dataset1
dataset1['day_of_week'] = dataset1.date_received.astype('str').apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
dataset1['day_of_month'] = dataset1.date_received.astype('str').apply(lambda x:int(x[6:8]))
dataset1['days_distance'] = dataset1.date_received.astype('str').apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,4,13)).days)
dataset1['discount_man'] = dataset1.discount_rate.apply(get_discount_man)
dataset1['discount_jian'] = dataset1.discount_rate.apply(get_discount_jian)
dataset1['is_man_jian'] = dataset1.discount_rate.apply(is_man_jian)
dataset1['discount_rate'] = dataset1.discount_rate.apply(calc_discount_rate)
d = dataset1[['coupon_id']]
d['coupon_count'] = 1
d = d.groupby('coupon_id').agg('sum').reset_index()
dataset1 = pd.merge(dataset1,d,on='coupon_id',how='left')
print('coupon_ralated1',dataset1.shape)
dataset1.to_csv('data/coupon1_feature.csv',index=None)
#dataset0
dataset0['day_of_week'] = dataset0.date_received.astype('str').apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
dataset0['day_of_month'] = dataset0.date_received.astype('str').apply(lambda x:int(x[6:8]))
dataset0['days_distance'] = dataset0.date_received.astype('str').apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,4,13)).days)
dataset0['discount_man'] = dataset0.discount_rate.apply(get_discount_man)
dataset0['discount_jian'] = dataset0.discount_rate.apply(get_discount_jian)
dataset0['is_man_jian'] = dataset0.discount_rate.apply(is_man_jian)
dataset0['discount_rate'] = dataset0.discount_rate.apply(calc_discount_rate)
d = dataset0[['coupon_id']]
d['coupon_count'] = 1
d = d.groupby('coupon_id').agg('sum').reset_index()
dataset0 = pd.merge(dataset0,d,on='coupon_id',how='left')
print('coupon_ralated0',dataset0.shape)
dataset0.to_csv('data/coupon0_feature.csv',index=None)



############# merchant related feature   #############
"""
1.merchant related: 
      total_sales. sales_use_coupon.  total_coupon
      coupon_rate = sales_use_coupon/total_sales.  
      transfer_rate = sales_use_coupon/total_coupon. 
      merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon

"""

#for dataset3
merchant3 = feature3[['merchant_id','coupon_id','distance','date_received','date']]

t = merchant3[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant3[pd.notnull(merchant3.date)][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant3[pd.notnull(merchant3.date)&pd.notnull(merchant3.coupon_id)][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant3[pd.notnull(merchant3.coupon_id)][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant3[pd.notnull(merchant3.date)&pd.notnull(merchant3.coupon_id)][['merchant_id','distance']]
t4.replace(np.nan,-1,inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1,np.nan,inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)

merchant3_feature = pd.merge(t,t1,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t2,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t3,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t5,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t6,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t7,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t8,on='merchant_id',how='left')
merchant3_feature.sales_use_coupon = merchant3_feature.sales_use_coupon.replace(np.nan,0) #fillna with 0
merchant3_feature['merchant_coupon_transfer_rate'] = merchant3_feature.sales_use_coupon.astype('float') / merchant3_feature.total_coupon
merchant3_feature['coupon_rate'] = merchant3_feature.sales_use_coupon.astype('float') / merchant3_feature.total_sales
merchant3_feature.total_coupon = merchant3_feature.total_coupon.replace(np.nan,0) #fillna with 0
print('merchant3',merchant3_feature.shape)
merchant3_feature.to_csv('data/merchant3_feature.csv',index=None)


#for dataset2
merchant2 = feature2[['merchant_id','coupon_id','distance','date_received','date']]

t = merchant2[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant2[pd.notnull(merchant2.date)][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant2[pd.notnull(merchant2.date)&pd.notnull(merchant2.coupon_id)][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant2[pd.notnull(merchant2.coupon_id)][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant2[pd.notnull(merchant2.date)&pd.notnull(merchant2.coupon_id)][['merchant_id','distance']]
t4.replace(np.nan,-1,inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1,np.nan,inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)

merchant2_feature = pd.merge(t,t1,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t2,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t3,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t5,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t6,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t7,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t8,on='merchant_id',how='left')
merchant2_feature.sales_use_coupon = merchant2_feature.sales_use_coupon.replace(np.nan,0) #fillna with 0
merchant2_feature['merchant_coupon_transfer_rate'] = merchant2_feature.sales_use_coupon.astype('float') / merchant2_feature.total_coupon
merchant2_feature['coupon_rate'] = merchant2_feature.sales_use_coupon.astype('float') / merchant2_feature.total_sales
merchant2_feature.total_coupon = merchant2_feature.total_coupon.replace(np.nan,0) #fillna with 0
print('merchant2',merchant2_feature.shape)
merchant2_feature.to_csv('data/merchant2_feature.csv',index=None)

#for dataset1
merchant1 = feature1[['merchant_id','coupon_id','distance','date_received','date']]

t = merchant1[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant1[pd.notnull(merchant1.date)][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant1[pd.notnull(merchant1.date)&pd.notnull(merchant1.coupon_id)][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant1[pd.notnull(merchant1.coupon_id)][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant1[pd.notnull(merchant1.date)&pd.notnull(merchant1.coupon_id)][['merchant_id','distance']]
t4.replace(np.nan,-1,inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1,np.nan,inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)


merchant1_feature = pd.merge(t,t1,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t2,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t3,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t5,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t6,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t7,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t8,on='merchant_id',how='left')
merchant1_feature.sales_use_coupon = merchant1_feature.sales_use_coupon.replace(np.nan,0) #fillna with 0
merchant1_feature['merchant_coupon_transfer_rate'] = merchant1_feature.sales_use_coupon.astype('float') / merchant1_feature.total_coupon
merchant1_feature['coupon_rate'] = merchant1_feature.sales_use_coupon.astype('float') / merchant1_feature.total_sales
merchant1_feature.total_coupon = merchant1_feature.total_coupon.replace(np.nan,0) #fillna with 0
print('merchant1',merchant1_feature.shape)
merchant1_feature.to_csv('data/merchant1_feature.csv',index=None)

#for dataset0
merchant0 = feature0[['merchant_id','coupon_id','distance','date_received','date']]

t = merchant0[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant0[pd.notnull(merchant0.date)][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant0[pd.notnull(merchant0.date)&pd.notnull(merchant0.coupon_id)][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant0[pd.notnull(merchant0.coupon_id)][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant0[pd.notnull(merchant0.date)&pd.notnull(merchant0.coupon_id)][['merchant_id','distance']]
t4.replace(np.nan,-1,inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1,np.nan,inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)


merchant0_feature = pd.merge(t,t1,on='merchant_id',how='left')
merchant0_feature = pd.merge(merchant0_feature,t2,on='merchant_id',how='left')
merchant0_feature = pd.merge(merchant0_feature,t3,on='merchant_id',how='left')
merchant0_feature = pd.merge(merchant0_feature,t5,on='merchant_id',how='left')
merchant0_feature = pd.merge(merchant0_feature,t6,on='merchant_id',how='left')
merchant0_feature = pd.merge(merchant0_feature,t7,on='merchant_id',how='left')
merchant0_feature = pd.merge(merchant0_feature,t8,on='merchant_id',how='left')
merchant0_feature.sales_use_coupon = merchant0_feature.sales_use_coupon.replace(np.nan,0) #fillna with 0
merchant0_feature['merchant_coupon_transfer_rate'] = merchant0_feature.sales_use_coupon.astype('float') / merchant0_feature.total_coupon
merchant0_feature['coupon_rate'] = merchant0_feature.sales_use_coupon.astype('float') / merchant0_feature.total_sales
merchant0_feature.total_coupon = merchant0_feature.total_coupon.replace(np.nan,0) #fillna with 0
print('merchant0',merchant0_feature.shape)
merchant0_feature.to_csv('data/merchant0_feature.csv',index=None)




############# user related feature   #############
"""
3.user related: 
      count_merchant. 
      user_avg_distance, user_min_distance,user_max_distance. 
      buy_use_coupon. buy_total. coupon_received.
      buy_use_coupon/coupon_received. 
      buy_use_coupon/buy_total
      user_date_datereceived_gap
"""

def get_user_date_datereceived_gap(s):
    s = s.split(':')
    return (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8])) - date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days

#for dataset3
user3 = feature3[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]

t = user3[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user3[pd.notnull(user3.date)][['user_id','merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id':'count_merchant'},inplace=True)

t2 = user3[pd.notnull(user3.date)&pd.notnull(user3.coupon_id)][['user_id','distance']]
t2.replace(np.nan,-1,inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1,np.nan,inplace=True)

t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance':'user_min_distance'},inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance':'user_max_distance'},inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance':'user_mean_distance'},inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance':'user_median_distance'},inplace=True)

t7 = user3[pd.notnull(user3.date)&pd.notnull(user3.coupon_id)][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user3[pd.notnull(user3.date)][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user3[pd.notnull(user3.coupon_id)][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user3[pd.notnull(user3.date_received)&pd.notnull(user3.date)][['user_id','date_received','date']]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id','user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap':'avg_user_date_datereceived_gap'},inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap':'min_user_date_datereceived_gap'},inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap':'max_user_date_datereceived_gap'},inplace=True)


user3_feature = pd.merge(t,t1,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t3,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t4,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t5,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t6,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t7,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t8,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t9,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t11,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t12,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t13,on='user_id',how='left')
user3_feature.count_merchant = user3_feature.count_merchant.replace(np.nan,0)
user3_feature.buy_use_coupon = user3_feature.buy_use_coupon.replace(np.nan,0)
user3_feature['buy_use_coupon_rate'] = user3_feature.buy_use_coupon.astype('float') / user3_feature.buy_total.astype('float')
user3_feature['user_coupon_transfer_rate'] = user3_feature.buy_use_coupon.astype('float') / user3_feature.coupon_received.astype('float')
user3_feature.buy_total = user3_feature.buy_total.replace(np.nan,0)
user3_feature.coupon_received = user3_feature.coupon_received.replace(np.nan,0)
print('user3',user3_feature.shape)
user3_feature.to_csv('data/user3_feature.csv',index=None)


#for dataset2
user2 = feature2[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]

t = user2[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user2[pd.notnull(user2.date)][['user_id','merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id':'count_merchant'},inplace=True)

t2 = user2[pd.notnull(user2.date)&pd.notnull(user2.coupon_id)][['user_id','distance']]
t2.replace(np.nan,-1,inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1,np.nan,inplace=True)

t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance':'user_min_distance'},inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance':'user_max_distance'},inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance':'user_mean_distance'},inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance':'user_median_distance'},inplace=True)

t7 = user2[pd.notnull(user2.date)&pd.notnull(user2.coupon_id)][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user2[pd.notnull(user2.date)][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user2[pd.notnull(user2.coupon_id)][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user2[pd.notnull(user2.date_received)&pd.notnull(user2.date)][['user_id','date_received','date']]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id','user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap':'avg_user_date_datereceived_gap'},inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap':'min_user_date_datereceived_gap'},inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap':'max_user_date_datereceived_gap'},inplace=True)

user2_feature = pd.merge(t,t1,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t3,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t4,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t5,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t6,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t7,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t8,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t9,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t11,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t12,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t13,on='user_id',how='left')
user2_feature.count_merchant = user2_feature.count_merchant.replace(np.nan,0)
user2_feature.buy_use_coupon = user2_feature.buy_use_coupon.replace(np.nan,0)
user2_feature['buy_use_coupon_rate'] = user2_feature.buy_use_coupon.astype('float') / user2_feature.buy_total.astype('float')
user2_feature['user_coupon_transfer_rate'] = user2_feature.buy_use_coupon.astype('float') / user2_feature.coupon_received.astype('float')
user2_feature.buy_total = user2_feature.buy_total.replace(np.nan,0)
user2_feature.coupon_received = user2_feature.coupon_received.replace(np.nan,0)
print('user2',user2_feature.shape)
user2_feature.to_csv('data/user2_feature.csv',index=None)


#for dataset1
user1 = feature1[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]

t = user1[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user1[pd.notnull(user1.date)][['user_id','merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id':'count_merchant'},inplace=True)

t2 = user1[pd.notnull(user1.date)&pd.notnull(user1.coupon_id)][['user_id','distance']]
t2.replace(np.nan,-1,inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1,np.nan,inplace=True)

t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance':'user_min_distance'},inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance':'user_max_distance'},inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance':'user_mean_distance'},inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance':'user_median_distance'},inplace=True)

t7 = user1[pd.notnull(user1.date)&pd.notnull(user1.coupon_id)][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user1[pd.notnull(user1.date)][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user1[pd.notnull(user1.coupon_id)][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user1[pd.notnull(user1.date_received)&pd.notnull(user1.date)][['user_id','date_received','date']]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id','user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap':'avg_user_date_datereceived_gap'},inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap':'min_user_date_datereceived_gap'},inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap':'max_user_date_datereceived_gap'},inplace=True)

user1_feature = pd.merge(t,t1,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t3,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t4,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t5,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t6,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t7,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t8,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t9,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t11,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t12,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t13,on='user_id',how='left')
user1_feature.count_merchant = user1_feature.count_merchant.replace(np.nan,0)
user1_feature.buy_use_coupon = user1_feature.buy_use_coupon.replace(np.nan,0)
user1_feature['buy_use_coupon_rate'] = user1_feature.buy_use_coupon.astype('float') / user1_feature.buy_total.astype('float')
user1_feature['user_coupon_transfer_rate'] = user1_feature.buy_use_coupon.astype('float') / user1_feature.coupon_received.astype('float')
user1_feature.buy_total = user1_feature.buy_total.replace(np.nan,0)
user1_feature.coupon_received = user1_feature.coupon_received.replace(np.nan,0)
print('user1',user1_feature.shape)
user1_feature.to_csv('data/user1_feature.csv',index=None)

#for dataset0
user0 = feature0[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]

t = user0[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user0[pd.notnull(user0.date)][['user_id','merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id':'count_merchant'},inplace=True)

t2 = user0[pd.notnull(user0.date)&pd.notnull(user0.coupon_id)][['user_id','distance']]
t2.replace(np.nan,-1,inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1,np.nan,inplace=True)

t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance':'user_min_distance'},inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance':'user_max_distance'},inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance':'user_mean_distance'},inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance':'user_median_distance'},inplace=True)

t7 = user0[pd.notnull(user0.date)&pd.notnull(user0.coupon_id)][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user0[pd.notnull(user0.date)][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user0[pd.notnull(user0.coupon_id)][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user0[pd.notnull(user0.date_received)&pd.notnull(user0.date)][['user_id','date_received','date']]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id','user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap':'avg_user_date_datereceived_gap'},inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap':'min_user_date_datereceived_gap'},inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap':'max_user_date_datereceived_gap'},inplace=True)

user0_feature = pd.merge(t,t1,on='user_id',how='left')
user0_feature = pd.merge(user0_feature,t3,on='user_id',how='left')
user0_feature = pd.merge(user0_feature,t4,on='user_id',how='left')
user0_feature = pd.merge(user0_feature,t5,on='user_id',how='left')
user0_feature = pd.merge(user0_feature,t6,on='user_id',how='left')
user0_feature = pd.merge(user0_feature,t7,on='user_id',how='left')
user0_feature = pd.merge(user0_feature,t8,on='user_id',how='left')
user0_feature = pd.merge(user0_feature,t9,on='user_id',how='left')
user0_feature = pd.merge(user0_feature,t11,on='user_id',how='left')
user0_feature = pd.merge(user0_feature,t12,on='user_id',how='left')
user0_feature = pd.merge(user0_feature,t13,on='user_id',how='left')
user0_feature.count_merchant = user0_feature.count_merchant.replace(np.nan,0)
user0_feature.buy_use_coupon = user0_feature.buy_use_coupon.replace(np.nan,0)
user0_feature['buy_use_coupon_rate'] = user0_feature.buy_use_coupon.astype('float') / user0_feature.buy_total.astype('float')
user0_feature['user_coupon_transfer_rate'] = user0_feature.buy_use_coupon.astype('float') / user0_feature.coupon_received.astype('float')
user0_feature.buy_total = user0_feature.buy_total.replace(np.nan,0)
user0_feature.coupon_received = user0_feature.coupon_received.replace(np.nan,0)
print('user0',user0_feature.shape)
user0_feature.to_csv('data/user0_feature.csv',index=None)


##################  user_merchant related feature #########################

"""
4.user_merchant:
      times_user_buy_merchant_before. 
"""
#for dataset3
all_user_merchant = feature3[['user_id','merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature3[['user_id','merchant_id','date']]
t = t[pd.notnull(t.date)][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature3[['user_id','merchant_id','coupon_id']]
t1 = t1[pd.notnull(t1.coupon_id)][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature3[['user_id','merchant_id','date','date_received']]
t2 = t2[pd.notnull(t2.date)&pd.notnull(t2.date_received)][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature3[['user_id','merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature3[['user_id','merchant_id','date','coupon_id']]
t4 = t4[pd.notnull(t4.date)&pd.isnull(t4.coupon_id)][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant3 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t1,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t2,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t3,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t4,on=['user_id','merchant_id'],how='left')
user_merchant3.user_merchant_buy_use_coupon = user_merchant3.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant3.user_merchant_buy_common = user_merchant3.user_merchant_buy_common.replace(np.nan,0)
user_merchant3['user_merchant_coupon_transfer_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype('float') / user_merchant3.user_merchant_received.astype('float')
user_merchant3['user_merchant_coupon_buy_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype('float') / user_merchant3.user_merchant_buy_total.astype('float')
user_merchant3['user_merchant_rate'] = user_merchant3.user_merchant_buy_total.astype('float') / user_merchant3.user_merchant_any.astype('float')
user_merchant3['user_merchant_common_buy_rate'] = user_merchant3.user_merchant_buy_common.astype('float') / user_merchant3.user_merchant_buy_total.astype('float')
print('user_merchant3',user_merchant3.shape)
user_merchant3.to_csv('data/user_merchant3.csv',index=None)

#for dataset2
all_user_merchant = feature2[['user_id','merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature2[['user_id','merchant_id','date']]
t = t[pd.notnull(t.date)][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature2[['user_id','merchant_id','coupon_id']]
t1 = t1[pd.notnull(t1.coupon_id)][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature2[['user_id','merchant_id','date','date_received']]
t2 = t2[pd.notnull(t2.date)&pd.notnull(t2.date_received)][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature2[['user_id','merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature2[['user_id','merchant_id','date','coupon_id']]
t4 = t4[pd.notnull(t4.date)&pd.isnull(t4.coupon_id)][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant2 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t1,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t2,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t3,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t4,on=['user_id','merchant_id'],how='left')
user_merchant2.user_merchant_buy_use_coupon = user_merchant2.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant2.user_merchant_buy_common = user_merchant2.user_merchant_buy_common.replace(np.nan,0)
user_merchant2['user_merchant_coupon_transfer_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype('float') / user_merchant2.user_merchant_received.astype('float')
user_merchant2['user_merchant_coupon_buy_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype('float') / user_merchant2.user_merchant_buy_total.astype('float')
user_merchant2['user_merchant_rate'] = user_merchant2.user_merchant_buy_total.astype('float') / user_merchant2.user_merchant_any.astype('float')
user_merchant2['user_merchant_common_buy_rate'] = user_merchant2.user_merchant_buy_common.astype('float') / user_merchant2.user_merchant_buy_total.astype('float')
print('user_merchant2',user_merchant2.shape)
user_merchant2.to_csv('data/user_merchant2.csv',index=None)

#for dataset1
all_user_merchant = feature1[['user_id','merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature1[['user_id','merchant_id','date']]
t = t[pd.notnull(t.date)][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature1[['user_id','merchant_id','coupon_id']]
t1 = t1[pd.notnull(t1.coupon_id)][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature1[['user_id','merchant_id','date','date_received']]
t2 = t2[pd.notnull(t2.date)&pd.notnull(t2.date_received)][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature1[['user_id','merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature1[['user_id','merchant_id','date','coupon_id']]
t4 = t4[pd.notnull(t4.date)&pd.isnull(t4.coupon_id)][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant1 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t1,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t2,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t3,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t4,on=['user_id','merchant_id'],how='left')
user_merchant1.user_merchant_buy_use_coupon = user_merchant1.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant1.user_merchant_buy_common = user_merchant1.user_merchant_buy_common.replace(np.nan,0)
user_merchant1['user_merchant_coupon_transfer_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype('float') / user_merchant1.user_merchant_received.astype('float')
user_merchant1['user_merchant_coupon_buy_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype('float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1['user_merchant_rate'] = user_merchant1.user_merchant_buy_total.astype('float') / user_merchant1.user_merchant_any.astype('float')
user_merchant1['user_merchant_common_buy_rate'] = user_merchant1.user_merchant_buy_common.astype('float') / user_merchant1.user_merchant_buy_total.astype('float')
print('user_merchant1',user_merchant1.shape)
user_merchant1.to_csv('data/user_merchant1.csv',index=None)


#for dataset0
all_user_merchant = feature0[['user_id','merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature0[['user_id','merchant_id','date']]
t = t[pd.notnull(t.date)][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature0[['user_id','merchant_id','coupon_id']]
t1 = t1[pd.notnull(t1.coupon_id)][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature0[['user_id','merchant_id','date','date_received']]
t2 = t2[pd.notnull(t2.date)&pd.notnull(t2.date_received)][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature0[['user_id','merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature0[['user_id','merchant_id','date','coupon_id']]
t4 = t4[pd.notnull(t4.date)&pd.isnull(t4.coupon_id)][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant0 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant0 = pd.merge(user_merchant0,t1,on=['user_id','merchant_id'],how='left')
user_merchant0 = pd.merge(user_merchant0,t2,on=['user_id','merchant_id'],how='left')
user_merchant0 = pd.merge(user_merchant0,t3,on=['user_id','merchant_id'],how='left')
user_merchant0 = pd.merge(user_merchant0,t4,on=['user_id','merchant_id'],how='left')
user_merchant0.user_merchant_buy_use_coupon = user_merchant0.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant0.user_merchant_buy_common = user_merchant0.user_merchant_buy_common.replace(np.nan,0)
user_merchant0['user_merchant_coupon_transfer_rate'] = user_merchant0.user_merchant_buy_use_coupon.astype('float') / user_merchant0.user_merchant_received.astype('float')
user_merchant0['user_merchant_coupon_buy_rate'] = user_merchant0.user_merchant_buy_use_coupon.astype('float') / user_merchant0.user_merchant_buy_total.astype('float')
user_merchant0['user_merchant_rate'] = user_merchant0.user_merchant_buy_total.astype('float') / user_merchant0.user_merchant_any.astype('float')
user_merchant0['user_merchant_common_buy_rate'] = user_merchant0.user_merchant_buy_common.astype('float') / user_merchant0.user_merchant_buy_total.astype('float')
print('user_merchant0',user_merchant0.shape)
user_merchant0.to_csv('data/user_merchant0.csv',index=None)











