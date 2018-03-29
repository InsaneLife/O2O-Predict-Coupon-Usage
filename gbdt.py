import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


dataset0 = pd.read_csv('data/dataset0.csv')
dataset0.fillna('0',inplace=True)
dataset0.label.replace(-1,0,inplace=True)
dataset1 = pd.read_csv('data/dataset1.csv')
dataset1.fillna('0',inplace=True)
dataset1.label.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('data/dataset2.csv')
dataset2.fillna('0',inplace=True)
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('data/dataset3.csv')
dataset3.fillna('0',inplace=True)

dataset0.drop_duplicates(inplace=True)
dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

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
print(dataset01_x.info())

####for params test
from sklearn.model_selection import GridSearchCV
param_test1 = {'min_samples_leaf':list(range(50,81,10))}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(
                                                               learning_rate=0.1,
                                                               n_estimators=100,
                                                               subsample=0.8,
                                                               max_features='sqrt',
                                                               max_depth=8,
                                                               min_samples_split=130,
                                                               # min_samples_leaf=30,
                                                               # min_weight_fraction_leaf=，
                                                               random_state=10,
                                                               ),
                        param_grid = param_test1,
                        scoring='roc_auc',
                        iid=False,
                        cv=5
)
gsearch1.fit(dataset01_x,dataset01_y)
print(gsearch1.cv_results_)
print(gsearch1.best_params_, gsearch1.best_score_)
print(roc_auc_score(dataset2_y,gsearch1.predict_proba(dataset2_x)[:,1]))

# ####for train
# gbm0 = GradientBoostingClassifier(max_features='sqrt',
#                                   learning_rate=0.1,
#                                   subsample=0.8,
#                                   random_state=10,
#                                   n_estimators=100,
#                                   min_samples_split=300,
#                                   max_depth=13,
#                                   min_samples_leaf=30
# )
# gbm0.fit(dataset01_x,dataset01_y)
# y_predprob = gbm0.predict_proba(dataset2_x)[:,1]
# print(roc_auc_score(dataset2_y,y_predprob))


# #for submit
# gbm0 = GradientBoostingClassifier(max_features='sqrt',
#                                   learning_rate=0.1,
#                                   min_samples_leaf=20,
#                                   subsample=0.8,
#                                   random_state=10,
#                                   n_estimators=100)
# gbm0.fit(dataset012_x,dataset012_y)
# dataset3_preds['label'] = gbm0.predict_proba(dataset3_x)[:,1]
# dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)
# dataset3_preds.to_csv("gbdt_preds.csv",index=None,header=None)
