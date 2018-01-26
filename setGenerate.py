import pandas as pd
import numpy as np
from datetime import date


##################  generate training and testing set ################
def get_label(s):
    s = s.split(':')
    if s[0]=='nan':
        return 0
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1
    else:
        return -1


coupon3 = pd.read_csv('data/coupon3_feature.csv')
merchant3 = pd.read_csv('data/merchant3_feature.csv')
user3 = pd.read_csv('data/user3_feature.csv')
user_merchant3 = pd.read_csv('data/user_merchant3.csv')
other_feature3 = pd.read_csv('data/other_feature3.csv')
online_feature3 = pd.read_csv('data/online_feature3.csv')
dataset3 = pd.merge(coupon3,merchant3,on='merchant_id',how='left')
dataset3 = pd.merge(dataset3,user3,on='user_id',how='left')
dataset3 = pd.merge(dataset3,user_merchant3,on=['user_id','merchant_id'],how='left')
dataset3 = pd.merge(dataset3,other_feature3,on=['user_id','coupon_id','date_received'],how='left')
dataset3 = pd.merge(dataset3,online_feature3,on='user_id',how='left')
dataset3.drop_duplicates(inplace=True)

dataset3.user_merchant_buy_total = dataset3.user_merchant_buy_total.replace(np.nan,0)
dataset3.user_merchant_any = dataset3.user_merchant_any.replace(np.nan,0)
dataset3.user_merchant_received = dataset3.user_merchant_received.replace(np.nan,0)
dataset3['is_weekend'] = dataset3.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset3.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset3 = pd.concat([dataset3,weekday_dummies],axis=1)
dataset3.drop(['merchant_id','day_of_week','coupon_count'],axis=1,inplace=True)
dataset3 = dataset3.replace('null',np.nan)
print(dataset3.shape)
dataset3.to_csv('data/dataset3.csv',index=None)


coupon2 = pd.read_csv('data/coupon2_feature.csv')
merchant2 = pd.read_csv('data/merchant2_feature.csv')
user2 = pd.read_csv('data/user2_feature.csv')
user_merchant2 = pd.read_csv('data/user_merchant2.csv')
other_feature2 = pd.read_csv('data/other_feature2.csv')
online_feature2 = pd.read_csv('data/online_feature2.csv')
dataset2 = pd.merge(coupon2,merchant2,on='merchant_id',how='left')
dataset2 = pd.merge(dataset2,user2,on='user_id',how='left')
dataset2 = pd.merge(dataset2,user_merchant2,on=['user_id','merchant_id'],how='left')
dataset2 = pd.merge(dataset2,other_feature2,on=['user_id','coupon_id','date_received'],how='left')
dataset2 = pd.merge(dataset2,online_feature2,on='user_id',how='left')
dataset2.drop_duplicates(inplace=True)

dataset2.user_merchant_buy_total = dataset2.user_merchant_buy_total.replace(np.nan,0)
dataset2.user_merchant_any = dataset2.user_merchant_any.replace(np.nan,0)
dataset2.user_merchant_received = dataset2.user_merchant_received.replace(np.nan,0)
dataset2['is_weekend'] = dataset2.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset2.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset2 = pd.concat([dataset2,weekday_dummies],axis=1)
dataset2['label'] = dataset2.date.astype('str') + ':' +  dataset2.date_received.astype('str')
dataset2.label = dataset2.label.apply(get_label)
dataset2.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)
dataset2 = dataset2.replace('null',np.nan)
print(dataset2.shape)
dataset2.to_csv('data/dataset2.csv',index=None)


coupon1 = pd.read_csv('data/coupon1_feature.csv')
merchant1 = pd.read_csv('data/merchant1_feature.csv')
user1 = pd.read_csv('data/user1_feature.csv')
user_merchant1 = pd.read_csv('data/user_merchant1.csv')
other_feature1 = pd.read_csv('data/other_feature1.csv')
online_feature1 = pd.read_csv('data/online_feature1.csv')
dataset1 = pd.merge(coupon1,merchant1,on='merchant_id',how='left')
dataset1 = pd.merge(dataset1,user1,on='user_id',how='left')
dataset1 = pd.merge(dataset1,user_merchant1,on=['user_id','merchant_id'],how='left')
dataset1 = pd.merge(dataset1,other_feature1,on=['user_id','coupon_id','date_received'],how='left')
dataset1 = pd.merge(dataset1,online_feature1,on='user_id',how='left')
dataset1.drop_duplicates(inplace=True)

dataset1.user_merchant_buy_total = dataset1.user_merchant_buy_total.replace(np.nan,0)
dataset1.user_merchant_any = dataset1.user_merchant_any.replace(np.nan,0)
dataset1.user_merchant_received = dataset1.user_merchant_received.replace(np.nan,0)
dataset1['is_weekend'] = dataset1.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset1.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset1 = pd.concat([dataset1,weekday_dummies],axis=1)
dataset1['label'] = dataset1.date.astype('str') + ':' +  dataset1.date_received.astype('str')
dataset1.label = dataset1.label.apply(get_label)
dataset1.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)
dataset1 = dataset1.replace('null',np.nan)
print(dataset1.shape)
dataset1.to_csv('data/dataset1.csv',index=None)


coupon0 = pd.read_csv('data/coupon0_feature.csv')
merchant0 = pd.read_csv('data/merchant0_feature.csv')
user0 = pd.read_csv('data/user0_feature.csv')
user_merchant0 = pd.read_csv('data/user_merchant0.csv')
other_feature0 = pd.read_csv('data/other_feature0.csv')
online_feature0 = pd.read_csv('data/online_feature0.csv')
dataset0 = pd.merge(coupon0,merchant0,on='merchant_id',how='left')
dataset0 = pd.merge(dataset0,user0,on='user_id',how='left')
dataset0 = pd.merge(dataset0,user_merchant0,on=['user_id','merchant_id'],how='left')
dataset0 = pd.merge(dataset0,other_feature0,on=['user_id','coupon_id','date_received'],how='left')
dataset0 = pd.merge(dataset0,online_feature0,on='user_id',how='left')
dataset0.drop_duplicates(inplace=True)

dataset0.user_merchant_buy_total = dataset0.user_merchant_buy_total.replace(np.nan,0)
dataset0.user_merchant_any = dataset0.user_merchant_any.replace(np.nan,0)
dataset0.user_merchant_received = dataset0.user_merchant_received.replace(np.nan,0)
dataset0['is_weekend'] = dataset0.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset0.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset0 = pd.concat([dataset0,weekday_dummies],axis=1)
dataset0['label'] = dataset0.date.astype('str') + ':' +  dataset0.date_received.astype('str')
dataset0.label = dataset0.label.apply(get_label)
dataset0.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)
dataset0 = dataset0.replace('null',np.nan)
print(dataset0.shape)
dataset0.to_csv('data/dataset0.csv',index=None)
