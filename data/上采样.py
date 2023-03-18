import numpy as np
import pandas as pd
 
 
def up_sample_data(df, percent=0.2):
    '''
    percent:少数类别样本数量的重采样的比例，可控制，一般不超过0.5，以免过拟合
    '''
    data1 = df[df['Label'] == 0]  # 将多数类别的样本放在data1
    data0 = df[df['Label'] == 1]  # 将少数类别的样本放在data0
    index = np.random.randint(
        len(data0), size= int(percent * (len(df) - len(data0))))  # 随机给定上采样取出样本的序号
    up_data0 = data0.iloc[list(index)]  # 上采样
    return(pd.concat([up_data0, data1]))
dataset = pd.read_csv('train_chuli.csv')
columns = ['PassengerId','Survived','Pclass','Sex','Age','SibSp',
'Parch','Fare','family_size','Embarked_C','Embarked_Q','Embarked_S']
