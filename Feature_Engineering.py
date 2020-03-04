# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 21:06
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : Feature_Engineering.py
import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import gc, os
import warnings

warnings.filterwarnings('ignore')


def get_weigth(x):
    if x in [0.6, 0.4]:
        return 1.9
    elif x in [0, 1]:
        return 1
    elif x in [0.2, 0.8]:
        return 1.5
    else:
        print('异常数据是:', x)
        raise NameError


# 将初赛数据和复赛数据进行合并
path = 'data/'
print('loading train dataset .......')
train_1 = pd.read_table(path + 'round1_iflyad_anticheat_traindata.txt')
train_2 = pd.read_table(path + 'round2_iflyad_anticheat_traindata.txt')
train = train_1.append(train_2).reset_index(drop=True)

print('loading test dataset ....')
test_a = pd.read_table(path + 'round2_iflyad_anticheat_testdata_feature_A.txt')
test_b = pd.read_table(path + 'round2_iflyad_anticheat_testdata_feature_B.txt')
test = test_a.append(test_b).reset_index(drop=True)
print('train:', train.shape, 'test:', test.shape)

print('Start data preprocessing')
if os.path.exists('catboost_0_proba.csv'):
    Submit_result = np.zeros((test.shape[0],))
    for index in range(5):
        test_proba = pd.read_csv('catboost_%d_proba.csv' % index, sep=',', usecols=['label'])
        Submit_result += np.array((test_proba['label'] * 5 >= 0.499).astype(int))

    test['weight'] = (Submit_result / 5)
    train['weight'] = train['label']

    train['weight'] = np.array(list(train['weight'].apply(lambda x: get_weigth(x))))
    test['weight'] = np.array(list(test['weight'].apply(lambda x: get_weigth(x))))

all_df = train.append(test).reset_index(drop=True)
del train_1, train_2, train, test_a, test_b

# 删除一些弱类别特征
del_cols = ['idfamd5', 'adidmd5', 'os']
for col in del_cols:
    all_df.drop(columns=col, inplace=True)

cat_list = [i for i in all_df.columns if i not in ['sid', 'label', 'nginxtime', 'ip', 'reqrealip']]


# 处理一下model 减少了3000左右的数据
def deal_model(x):
    if '-' in x:
        x = x.replace('-', ' ')
    if '%20' in x:
        x = x.replace('%20', ' ')
    if '_' in x:
        x = x.replace('_', ' ')
    if '+' in x:
        x = x.replace('+', ' ')
    if 'the palace museum edition' in x:
        x = 'mix 3'
    return x


print('loading model ........')
all_df['model'] = all_df['model'].astype('str').map(lambda x: x.lower()).apply(deal_model)


# 对于model 选择model排名前1200数据
models = []
for i in all_df['model'].value_counts().head(1200).index:
    models.append(i)
all_df.loc[~all_df['model'].isin(models), 'model'] = 'others'


# 这个版本是最后的版本 不在进行任何的修改操作
def my_ver_trans(x):
    x = str(x).replace('[', '').replace(']', '').replace('".', '').replace('"', '')

    if x[0:3] == '309':
        if len(x) > 4:
            return '3.9.' + x[3:5] + '.' + x[5:7]
        else:
            return '3.9.' + '0.0.0'

    if x[0:3] == '190':
        if len(x) > 4:
            return '1.9.' + x[3] + '.' + x[4:]
        else:
            return '1.9.' + '0.0'

    if '521000' in x or '5.2.1' in x:
        return '5.2.1.0'

    # 处理全部都是数值数据
    if '.' not in x:
        if len(x) >= 3:
            return x[0] + '.' + x[1] + '.' + x[2] + '.' + x[3:]
        elif len(x) > 1 and len(x) < 3:
            return x[0] + '.' + x[1] + '.0'
        else:
            return x + '.0.0.0'

    # 处理带点 但是数据位数小于三位的情况
    tmp = x.split('.')
    if len(tmp) < 3:
        return x + '.0.0'

    return x


print('loading ver .............')
all_df['ver'] = all_df.ver.apply(my_ver_trans)

# 得到ver的前三位
lst3 = []
for val in all_df['ver'].values:
    val = str(val).split('.')
    if len(val) < 3:
        val.append('0')
        val.append('0')
    lst3.append(val[0] + '.' + val[1] + '.' + val[2])
all_df['ver3'] = lst3


# 对osv进行清洗操作
def my_osv_trans(x):
    # 首先处理数据中带有 android以及v 的数据
    x = str(x).lower().replace('android_', '').replace('android', '').replace('v', '').replace('_', '.').replace(
        '-', '.')

    # 因为不知道44是不是表示一个类别 暂时不使用
    if '.' not in x:
        if len(x) >= 3:
            return x[0] + '.' + x[1] + '.' + x[2] + '.' + x[3:]
        elif len(x) > 1 and len(x) < 3:
            return x[0] + '.' + x[1] + '.0'
        else:
            return x + '.0.0.0'

    if '.' not in x:
        return x[0] + '.0.0.0.0'
    # 处理带点 但是数据位数小于三位的情况
    tmp = x.split('.')
    if len(tmp) < 3:
        return x + '.0.0.0'

    return x


print('loading osv .............')
all_df['osv'] = all_df.osv.apply(my_osv_trans)

# 得到osv的前三个值
lst3 = []
for val in all_df['osv'].values:
    if 'nan' in val:
        val = '0.0.0.0'

    val = str(val).split('.')
    if len(val) < 3:
        val.append('0')
        val.append('0')
    lst3.append(val[0] + '.' + val[1] + '.' + val[2])

all_df['osv3'] = lst3


# 处理lan 这一版本 暂时修改
def lan(x):
    x = x.replace('_', '-')
    if x == '-cn' or x == 'ko-cn':
        return 'cn'
    if 'vi' in x:
        return 'vi'
    if 'zh-' == x:
        return 'zh'
    return x


print('loading lan ..............')
all_df['lan'] = all_df['lan'].astype(str).map(lambda x: x.lower()).apply(lan)

# 处理lan
lans = []
for i in all_df['lan'].value_counts().head(12).index:
    lans.append(i)
all_df.loc[~all_df['lan'].isin(lans), 'lan'] = 'others'

# 构建特征工程
print('Build Feature Engineering')
all_df['time'] = pd.to_datetime(all_df['nginxtime']*1e+6) + timedelta(hours=8)
all_df['day'] = all_df['time'].dt.day
all_df['hour'] = all_df['time'].dt.hour

all_df['size'] = (np.sqrt(all_df['h']**2 + all_df['w'] ** 2) / 2.54) / 1000
all_df['ratio'] = all_df['h'] / all_df['w']
all_df['px'] = all_df['ppi'] * all_df['size']
all_df['mj'] = all_df['h'] * all_df['w']

# 处理ip数据
all_df['ip_0'] = all_df['ip'].map(lambda x: '.'.join(x.split('.')[:1]))
all_df['ip_1'] = all_df['ip'].map(lambda x: '.'.join(x.split('.')[0:2]))
all_df['ip_2'] = all_df['ip'].map(lambda x: '.'.join(x.split('.')[0:3]))

all_df['reqrealip_0'] = all_df['reqrealip'].map(lambda x: '.'.join(str(x).split('.')[:1]))
all_df['reqrealip_1'] = all_df['reqrealip'].map(lambda x: '.'.join(str(x).split('.')[0:2]))
all_df['reqrealip_2'] = all_df['reqrealip'].map(lambda x: '.'.join(str(x).split('.')[0:3]))

all_df['ip_equal'] = (all_df['ip'] == all_df['reqrealip']).astype(int)

ip_feat = ['ip_0', 'ip_1', 'ip_2', 'reqrealip_0', 'reqrealip_1', 'reqrealip_2', 'ip_equal']

print('********执行交叉特征********')
all_df['device'] = all_df['model'].astype(str) + all_df['make'].astype(str) + all_df['osv'].astype(str) + all_df['lan'].astype(str) + all_df['h'].astype(str) + all_df['w'].astype(str) + all_df['ppi'].astype(str)

cross_feature = []
col_name1 = "cross_" + 'pkgname' + "_and_" + 'adunitshowid'
all_df[col_name1] = all_df['pkgname'].astype(str) + '_' + all_df['adunitshowid'].astype(str)
cross_feature.append(col_name1)
print('ADD 1 Done!')
col_name2 = "cross_" + 'pkgname' + "_and_" + 'apptype'
all_df[col_name2] = all_df['pkgname'].astype(str) + '_' + all_df['model'].astype(str)
cross_feature.append(col_name2)
print('ADD 2 Done!')
col_name3 = "cross_" + 'pkgname' + "_and_" + 'model'
all_df[col_name3] = all_df['pkgname'].astype(str) + '_' + all_df['model'].astype(str)
cross_feature.append(col_name3)
print('ADD 3 Done!')
col_name4 = "cross_" + 'imeimd5' + "_and_" + 'h'
all_df[col_name4] = all_df['imeimd5'].astype(str) + '_' + all_df['h'].astype(str)
cross_feature.append(col_name4)
print('ADD 4 Done!')
col_name5 = "cross_" + 'model' + "_and_" + 'osv'
all_df[col_name5] = all_df['model'].astype(str) + '_' + all_df['osv'].astype(str)
cross_feature.append(col_name5)
print('ADD 5 Done!')
col_name6 = "cross_" + 'device' + "_and_" + 'imeimd5'
all_df[col_name6] = all_df['device'].astype(str) + '_' + all_df['imeimd5'].astype(str)
cross_feature.append(col_name6)
print('ADD 6 Done!')
col_name7 = "cross_" + 'dvctype' + "_and_" + 'model'
all_df[col_name7] = all_df['dvctype'].astype(str) + '_' + all_df['model'].astype(str)
cross_feature.append(col_name7)
print('ADD 7 Done!')

object_col = [i for i in all_df.select_dtypes(object).columns if i not in ['sid', 'label']]
for i in tqdm(object_col):
    lbl = LabelEncoder()
    all_df[i] = lbl.fit_transform(all_df[i].astype(str))

cat_list = cat_list + ['device', 'hour', 'day'] + ip_feat + cross_feature
for i in tqdm(cat_list):
    all_df['{}_count'.format(i)] = all_df.groupby(['{}'.format(i)])['sid'].transform('count')
print('cat_list:', cat_list)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


print('reduce_mem_usage ............')
all_df = reduce_mem_usage(all_df)

all_df.to_csv('all_data.csv', index=False)
print('Save data done! ............')
