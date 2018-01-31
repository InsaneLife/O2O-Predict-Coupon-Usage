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
# t1 = feature3[pd.notnull(feature3.coupon_id)][['user_id']]
# t1['online_coupon_receive'] = 1
# t1['online_coupon_receive'] = t1.groupby('user_id').agg('sum').reset_index()

# online_feature3 = pd.merge(t,t1,on='user_id',how='outer')
# print(online_feature3.describe())
# online_feature3.to_csv('data/online_feature3.csv',index=None)

#for dataset2
t = feature2[pd.notnull(feature2.date)][['user_id']]
t['online_total_consume'] = 1
t['online_total_consume'] = t.groupby('user_id').agg('sum').reset_index()
t.to_csv('data/online_feature2.csv',index=None)
# t1 = feature2[pd.notnull(feature2.coupon_id)][['user_id']]
# t1['online_coupon_receive'] = 1
# t1['online_coupon_receive'] = t1.groupby('user_id').agg('sum').reset_index()
#
# online_feature2 = pd.merge(t,t1,on='user_id',how='outer')
# print(online_feature2.describe())
# online_feature2.to_csv('data/online_feature2.csv',index=None)


#for dataset1
t = feature1[pd.notnull(feature1.date)][['user_id']]
t['online_total_consume'] = 1
t['online_total_consume'] = t.groupby('user_id').agg('sum').reset_index()
t.to_csv('data/online_feature1.csv',index=None)
# t1 = feature1[pd.notnull(feature1.coupon_id)][['user_id']]
# t1['online_coupon_receive'] = 1
# t1['online_coupon_receive'] = t1.groupby('user_id').agg('sum').reset_index()
#
# online_feature1 = pd.merge(t,t1,on='user_id',how='outer')
# print(online_feature1.describe())
# online_feature1.to_csv('data/online_feature1.csv',index=None)

#for dataset0
t = feature0[pd.notnull(feature0.date)][['user_id']]
t['online_total_consume'] = 1
t['online_total_consume'] = t.groupby('user_id').agg('sum').reset_index()
t.to_csv('data/online_feature0.csv',index=None)
# t1 = feature0[pd.notnull(feature0.coupon_id)][['user_id']]
# t1['online_coupon_receive'] = 1
# t1['online_coupon_receive'] = t1.groupby('user_id').agg('sum').reset_index()
#
# online_feature0 = pd.merge(t,t1,on='user_id',how='outer')
# print(online_feature0.describe())
# online_feature0.to_csv('data/online_feature0.csv',index=None)