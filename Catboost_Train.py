# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 20:08
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Catboost_Train.py
import pandas as pd
import gc
import math
import catboost as cbt
import numpy as np
from sklearn.model_selection import StratifiedKFold

path = './'

# 测试集的sid
test = pd.read_csv(path + 'aichallenge2019_test_all_sample.csv', sep=',', usecols=['sid'])

all_data = pd.read_csv(path + 'all_data.csv', sep=',')

cat_list = ['adunitshowid', 'apptype', 'carrier', 'city', 'dvctype', 'h', 'imeimd5', 'lan', 'macmd5', 'make', 'mediashowid', 'model', 'ntt', 'openudidmd5', 'orientation', 'osv', 'pkgname', 'ppi', 'province', 'ver', 'w', 'device', 'hour', 'day', 'ip_0', 'ip_1', 'ip_2', 'reqrealip_0', 'reqrealip_1', 'reqrealip_2', 'ip_equal', 'cross_pkgname_and_adunitshowid', 'cross_pkgname_and_apptype', 'cross_pkgname_and_model', 'cross_imeimd5_and_h', 'cross_model_and_osv', 'cross_device_and_imeimd5', 'cross_dvctype_and_model']

feature_name = [i for i in all_data.columns if i not in ['sid', 'label', 'time']]


# 划分训练集以及测试集
tr_index = ~all_data['label'].isnull()
X_train = all_data[tr_index][list(set(feature_name))].reset_index(drop=True)
y = all_data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = all_data[~tr_index][list(set(feature_name))].reset_index(drop=True)
print('X_train.shape, X_test.shape:', X_train.shape, X_test.shape)

n_split = 5
random_seed = 2019
pass_train = False
Submit_result = np.zeros((X_test.shape[0],))

del all_data

skf = StratifiedKFold(n_splits=n_split, random_state=random_seed, shuffle=True)
for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
    print(index, train_index[0:5], test_index[0:5])

    # 如果需要接着训练的话 则跳过的训练次数
    if pass_train:
        if index < 1:
            # print('pass %d:' % index, train_index[0:5], test_index[0:5])
            continue

    train_x, test_x, train_y, test_y = X_train[feature_name].iloc[train_index], X_train[feature_name].iloc[test_index], \
                                       y.iloc[train_index], y.iloc[test_index]

    # cat_features是指 用来做处理的类别特征
    cbt_model = cbt.CatBoostClassifier(iterations=5000, learning_rate=0.05, max_depth=11, l2_leaf_reg=1, verbose=10,
                                       early_stopping_rounds=400, task_type='GPU', eval_metric='F1',
                                       cat_features=cat_list)

    cbt_model.fit(train_x[feature_name], train_y, eval_set=(test_x[feature_name], test_y))

    # 训练完成之后保存一下模型
    del train_x, train_y, test_x, test_y
    gc.collect()

    # catboost测试数据的时候使用的是CPU相对较慢 所以建议使用分批预测模型
    num = 40
    line_num = math.floor(len(X_test) / num)
    print('line_num:', line_num)

    Proba_result = []
    for i in range(num):
        test_1 = X_test.loc[i * line_num:(i + 1) * line_num - 1, feature_name]
        test_pred = cbt_model.predict_proba(test_1)[:, 1] / n_split
        Proba_result.extend(list(test_pred))

    Proba_result = np.array(Proba_result)
    Submit_result += Proba_result

    print('Saving result proba:', index)
    submit = test[['sid']]
    submit['label'] = Proba_result
    submit.to_csv('catboost_%d_proba.csv' % index, index=False)

    # 继续释放内存
    del cbt_model, Proba_result
    gc.collect()

submit_1 = test[['sid']]
submit_1['label'] = (Submit_result >= 0.499).astype(int)
print('生成结果的正负比:\n', submit_1['label'].value_counts())
submit_1.to_csv("submit.csv", index=False)
