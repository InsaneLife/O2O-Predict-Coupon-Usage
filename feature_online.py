import pandas as pd
import numpy as np
from datetime import date
pd.options.mode.chained_assignment = None  # default='warn'

online_train = pd.read_csv('data/ccf_online_stage1_train.csv',header=0,dtype=str)
online_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']


feature3 = online_train[((online_train.date>='20160415')&(online_train.date<='20160630'))|((pd.isnull(online_train.date))&(online_train.date_received>='20160415')&(online_train.date_received<='20160630'))]
feature2 = online_train[(online_train.date>='20160301')&(online_train.date<='20160514')|((pd.isnull(online_train.date))&(online_train.date_received>='20160301')&(online_train.date_received<='20160514'))]
feature1 = online_train[(online_train.date>='20160201')&(online_train.date<='20160413')|((pd.isnull(online_train.date))&(online_train.date_received>='20160201')&(online_train.date_received<='20160413'))]
feature0 = online_train[((online_train.date>='20160101')&(online_train.date<='20160313'))|((pd.isnull(online_train.date))&(online_train.date_received>='20160101')&(online_train.date_received<='20160313'))]

"""
    t:线上成功消费次数
    t1:线上获得消费券次数
"""

#for dataset3
t = feature3[pd.notnull(feature3.date)][['user_id']]
t['online_total_consume'] = 1
t['online_total_consume'] = t.groupby('user_id').agg('sum').reset_index()
t.to_csv('data/online_feature3.csv',index=None)
print('online3',feature3.shape)

#for dataset2
t = feature2[pd.notnull(feature2.date)][['user_id']]
t['online_total_consume'] = 1
t['online_total_consume'] = t.groupby('user_id').agg('sum').reset_index()
t.to_csv('data/online_feature2.csv',index=None)
print('online2',feature2.shape)


#for dataset1
t = feature1[pd.notnull(feature1.date)][['user_id']]
t['online_total_consume'] = 1
t['online_total_consume'] = t.groupby('user_id').agg('sum').reset_index()
t.to_csv('data/online_feature1.csv',index=None)
print('online1',feature1.shape)

#for dataset0
t = feature0[pd.notnull(feature0.date)][['user_id']]
t['online_total_consume'] = 1
t['online_total_consume'] = t.groupby('user_id').agg('sum').reset_index()
t.to_csv('data/online_feature0.csv',index=None)
print('online0',feature0.shape)