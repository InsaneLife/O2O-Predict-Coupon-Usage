import pandas as pd

xgb = pd.read_csv('xgb_preds.csv')
xgb.columns = ['user_id','coupon_id','date','xgb_preds']
xgb.drop_duplicates(['user_id','coupon_id','date'],inplace=True)
# print(xgb.info())
gbdt = pd.read_csv('gbdt_preds.csv')
gbdt.columns = ['user_id','coupon_id','date','gbdt_preds']
gbdt.drop_duplicates(['user_id','coupon_id','date'],inplace=True)
# print(gbdt.info())

blending = pd.merge(xgb,gbdt,on=['user_id','coupon_id','date'],how='inner')
blending['preds'] = blending['xgb_preds']*0.6 + blending['gbdt_preds']*0.4
blending.drop(['xgb_preds','gbdt_preds'],axis=1,inplace=True)

blending.to_csv('blending_preds.csv',index=None,header=None)