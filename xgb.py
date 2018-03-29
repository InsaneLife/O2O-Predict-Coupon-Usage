import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

dataset0 = pd.read_csv('data/dataset0.csv')
dataset0.label.replace(-1,0,inplace=True)
dataset1 = pd.read_csv('data/dataset1.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('data/dataset2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('data/dataset3.csv')

dataset0.drop_duplicates(inplace=True)
dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)
print(dataset3.info())

dataset01 = pd.concat([dataset0,dataset1],axis=0)
dataset012 = pd.concat([dataset01,dataset2],axis=0)

dataset01_y = dataset01.label
dataset01_x = dataset01.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)  # 'day_gap_before','day_gap_after' cause overfitting, 0.77
dataset2_y = dataset2.label
dataset2_x = dataset2.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset012_y = dataset012.label
dataset012_x = dataset012.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
dataset3_preds = dataset3[['user_id','coupon_id','date_received']]
dataset3_x = dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after'],axis=1)

print(dataset01_x.shape,dataset2_x.shape,dataset3_x.shape)

dataset01 = xgb.DMatrix(dataset01_x,label=dataset01_y)
dataset2 = xgb.DMatrix(dataset2_x,label=dataset2_y)
dataset012 = xgb.DMatrix(dataset012_x,label=dataset012_y)
dataset3 = xgb.DMatrix(dataset3_x)

#for params search
from sklearn.model_selection import GridSearchCV
"""
	1.学习率、决策树数量（'gamma':0.1,）
	2.单决策树相关参数(max_depth, min_child_weight, gamma, subsample, colsample_bytree,'colsample_bylevel':0.65,)
	3.正则化调优(lambda, alpha)
"""
param_test = {}

#
# params={'booster':'gbtree',
# 	    'gamma':0.1,
# 	    'min_child_weight':1.1,
# 	    'max_depth':5,
# 	    'lambda':10,
# 	    'subsample':0.65,
# 	    'colsample_bytree':0.65,
# 	    'colsample_bylevel':0.65,
# 	    'eta': 0.01,
# 	    'tree_method':'exact',
# 	    'seed':0,
# 	    'nthread':12
# 	    }

# # train on dataset01, evaluate on dataset2
# from sklearn.metrics import roc_auc_score
# watchlist = [(dataset01,'train')]
# model = xgb.train(params,dataset01,num_boost_round=2000,evals=watchlist)
# preds = model.predict(dataset2)
# preds = MinMaxScaler().fit_transform(preds.reshape(-1,1))
# print(roc_auc_score(dataset2.get_label(),preds))


# # for submit
# watchlist = [(dataset012,'train')]
# model = xgb.train(params,dataset012,num_boost_round=2000,evals=watchlist)
#
# #predict test set
# dataset3_preds['label'] = model.predict(dataset3)
# dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label.reshape(-1,1))
# dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)
# dataset3_preds.to_csv("xgb_preds.csv",index=None,header=None)
# # print(dataset3_preds.info())
#
# #
# #save feature score
# feature_score = model.get_fscore()
# feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
# fs = []
# for (key,value) in feature_score:
#     fs.append("{0},{1}\n".format(key,value))
#
# with open('xgb_feature_score.csv','w') as f:
#     f.writelines("feature,score\n")
#     f.writelines(fs)
#
