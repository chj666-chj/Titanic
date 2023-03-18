import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_ROC(fpr, tpr, auc, path):
    plt.figure(figsize=(8, 7), dpi=80, facecolor='w')  # dpi:每英寸长度的像素点数；facecolor 背景颜色
    plt.xlim((-0.01, 1.02))  # x,y 轴刻度的范围
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))  # 绘制刻度
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.plot(fpr, tpr, 'r-', lw=2, label='AUC=%.4f' % auc)  # 绘制AUC 曲线
    plt.legend(loc='lower right')  # 设置显示标签的位置
    plt.xlabel('False Positive Rate', fontsize=14)  # 绘制x,y 坐标轴对应的标签
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.grid(b=True, ls=':')  # 绘制网格作为底板;b是否显示网格线；ls表示line style
    plt.title(u'TabTransformer ROC curve And  AUC', fontsize=18)  # 打印标题
    roc_image_path = os.path.join(path, "post_ROC.svg")
    plt.savefig(roc_image_path, format="svg")
    plt.close('all')
    # plt.show()

# Helper method to print importances and visualize distribution
# ------------------------------------------------------------#
# plt.bar(x，height， width，*，align=‘center’，**kwargs):绘制柱状图
# x:包含所有柱子的下标的列表
# height:包含所有柱子的高度值的列表
# align:柱子对齐方式，有两个可选值：center和edge。center表示每根柱子是根据下标来对齐, edge则表示每根柱子全部以下标为起点，然后显示到下标的右边
# plt.xticks():获取或设置当前 x 轴或 y 轴刻度位置和标签（即设置 x 或 y 轴的标 签）
# plt.xticks([0,1],[1,2],rotation=0)
# [0,1]代表x坐标轴的0和1位置，[2,3]代表0,1位置的显示lable，rotation代表lable显示的旋转角度。
# --------------------------------------------------------------#
def Separate_importances(feature_names, path, importances, title="Average Feature Importances",
                          axis_title="Features", fontsize=15):
    plt.close('all')
    # print(len(feature_names))
    x_pos = np.arange(len(feature_names))
    # x_pos = np.arange(len(feature_names)-1)
    print('x_pos.shape', x_pos.shape)
    # plt.figure(figsize=(8, 7), dpi=80, facecolor='w')  # dpi:每英寸长度的像素点数；facecolor 背景颜色
    plt.figure(figsize=(43, 6))                  # figsize=(43, 6), dpi=600
    print(importances.shape)
    importances = importances.reshape(-1,)
    plt.bar(x_pos, importances, align='center')  # width=0.5
    plt.xticks(x_pos, feature_names, fontsize=12, rotation=90)
    # plt.xticks(x_pos, feature_names[:-1], fontsize=12, rotation=90)

    plt.xlabel(axis_title)
    plt.title(title)
    for a, b in zip(x_pos, importances):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=fontsize)  # 每个柱形图显示数值

    Separate_importances_image_path = os.path.join(path, "Separate_importances.svg")
    plt.savefig(Separate_importances_image_path, format="svg")
    plt.show()

def Combined_importances(importances, path):
    plt.clf()
    df = pd.DataFrame({'Lactate':    [importances[0], 0, 0, 0, 0, 0],
                       'pH':         [importances[1], 0, 0, 0, 0, 0],
                       'Anion Gap':  [importances[2], 0, 0, 0, 0, 0],
                       'Bicarbonate':[importances[3], 0, 0, 0, 0, 0],
                       'Calcium':    [importances[4], 0, 0, 0, 0, 0],
                       'Chloride':   [importances[5], 0, 0, 0, 0, 0],
                       'Glucose':    [importances[6], 0, 0, 0, 0, 0],
                       'Potassium':  [importances[7], 0, 0, 0, 0, 0],
                       'Sodium':     [importances[8], 0, 0, 0, 0, 0],
                       'Hematocrit': [importances[9], 0, 0, 0, 0, 0],
                       'Hemoglobin': [importances[10], 0, 0, 0, 0, 0],

                       'Creatinine':   [0, importances[11], 0, 0, 0, 0],
                       'UreaNitogen':  [0, importances[12], 0, 0, 0, 0],
                       'Magnesium':    [0, importances[13], 0, 0, 0, 0],
                       'Phosphate':    [0, importances[14], 0, 0, 0, 0],

                       'PT':        [0, 0, importances[15], 0, 0, 0],
                       'PTT':       [0, 0, importances[16], 0, 0, 0],
                       'INR(PT)':   [0, 0, importances[17], 0, 0, 0],

                       'Lymphocytes':        [0, 0, 0, importances[18], 0, 0],
                       'Monocytes':          [0, 0, 0, importances[19], 0, 0],
                       'Neutrophil':         [0, 0, 0, importances[20], 0, 0],
                       'White_Blood_Cells':  [0, 0, 0, importances[21], 0, 0],
                       'Basophils':          [0, 0, 0, importances[22], 0, 0],
                       'Eosinophils':        [0, 0, 0, importances[23], 0, 0],
                       'Platelet_Count':     [0, 0, 0, importances[24], 0, 0],

                       'AlanineAminotransferase(ALT)':  [0, 0, 0, 0, importances[25], 0],
                       'AlkalinePhosphatase':           [0, 0, 0, 0, importances[26], 0],
                       'AspirateAminotransferase(AST)': [0, 0, 0, 0, importances[27], 0],
                       'Albumin':                       [0, 0, 0, 0, importances[28], 0],
                       'Bilirubin':                     [0, 0, 0, 0, importances[29], 0],

                       'MCH':              [0, 0, 0, 0, 0, importances[30]],
                       'MCHC':             [0, 0, 0, 0, 0, importances[31]],
                       'MCV':              [0, 0, 0, 0, 0, importances[32]],
                       'RDW':              [0, 0, 0, 0, 0, importances[33]],
                       'Red _Blood_Cells': [0, 0, 0, 0, 0, importances[34]],
                       'patients_age':     [0, 0, 0, 0, 0, importances[35]],
                       })
    print('df.shape', df.shape)
    print('df.type', type(df))

    cols = ['dongwuqixue', 'shengongneng', 'ningxuegongneng', 'yanzhengmianyi', 'gangongneng', 'qita']
    df.index = cols
    plt.figure(figsize=(43, 6))
    df = df.astype(float)
    df.plot.bar(stacked=True, rot=30, figsize=(10, 6))   # rot 旋转刻度标签(0-360)
    plt.legend(prop={'size':5},loc=1)

    Combined_importances_image_path = os.path.join(path, " Combined_importances.svg")
    plt.savefig(Combined_importances_image_path, format="svg")
    plt.show()

def stander_data(data):
    '''
    标准化
    :return:
    '''
    # data = pd.read_csv('./data/pyh_three_discreate.csv').sample(frac=1)
    # data1=data.iloc[:,:29]
    # data2=data.iloc[:,29:]
    # # # print(data1[:10])
    # # # print(data2[:10])
    # data1 = data1.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    # # # data1 = data1.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    # data = pd.concat([data1, data2], axis=1)
    # print(data[:10])

    data = data.iloc[:, :]
    data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    return data

class TabDataset(Dataset):
    def __init__(self, data, target=None):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        _dict = {'data': torch.FloatTensor(data)}

        # data = self.data[idx]
        # # # print(type(data))  #<class 'numpy.ndarray'>
        # _dict = {'data': torch.tensor(data, dtype=torch.float)}

        if self.target is not None:
            target = self.target[idx].item()
            _dict.update({'target': torch.tensor(target, dtype=torch.float)})
        return _dict

def mapping_dataloaders(path:str,batch_size,feature_names):
    np.random.seed(256)
    df = pd.read_csv(path,header=None)  # 筛选后的训练/测试集，无表头
    print(df[:10])
    print('列名：',[column for column in df])

    df.columns = [i for i in feature_names]
    print(df[:10])
    print('列名：', [column for column in df])
    labels = df.pop('yfx')  # 取出标签列
    data = df.to_numpy()   # df->>numpy
    print('all_data.shape:', data.shape)

    dataset = TabDataset(data=data, target=None)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataset

# 读取原始数据
def get_data(path: str, logs):
    np.random.seed(256)
    pre_pyh_df = pd.read_csv(path).sample(frac=1)  # 原始数据，有表头
    # pre_pyh_df = pd.read_csv(path)
    pre_pyh_label = pre_pyh_df.pop('Survived').to_numpy()

    pre_pyh_train_labels = torch.LongTensor(pre_pyh_label[:int(len(pre_pyh_df) * 0.7)])  # numpy->>>torch.tensor,int64
    pre_pyh_test_labels = torch.LongTensor(pre_pyh_label[int(len(pre_pyh_df) * 0.7):])  # numpy->>>torch.tensor,int64

    # # 'dongwuqixue'第一类,
    # Lactate     = pre_pyh_df.pop('Lactate')
    # pH          = pre_pyh_df.pop('pH')
    # Anion_Gap   = pre_pyh_df.pop('Anion Gap')
    # Bicarbonate = pre_pyh_df.pop('Bicarbonate')
    # Calcium     = pre_pyh_df.pop('Calcium')
    # Chloride    = pre_pyh_df.pop('Chloride')
    # Glucose     = pre_pyh_df.pop('Glucose')
    # Potassium   =  pre_pyh_df.pop('Potassium')
    # Sodium      = pre_pyh_df.pop('Sodium')
    # Hematocrit  = pre_pyh_df.pop('Hematocrit')
    # Hemoglobin  = pre_pyh_df.pop('Hemoglobin')

    # # 'shengongneng'第二类,
    # Creatinine  = pre_pyh_df.pop('Creatinine')
    # UreaNitogen = pre_pyh_df.pop('UreaNitogen')
    # Magnesium   = pre_pyh_df.pop('Magnesium')
    # Phosphate   = pre_pyh_df.pop('Phosphate')
    #
    # # 'ningxuegongneng'第三类,
    # PT      = pre_pyh_df.pop('PT')
    # PTT     = pre_pyh_df.pop('PTT')
    # INR = pre_pyh_df.pop('INR(PT)')
    #
    # # 'yanzhengmianyi'第四类,
    # Lymphocytes       = pre_pyh_df.pop('Lymphocytes')
    # Monocytes         = pre_pyh_df.pop('Monocytes')
    # Neutrophil        = pre_pyh_df.pop('Neutrophil')
    # White_Blood_Cells = pre_pyh_df.pop('White_Blood_Cells')
    # Basophils         = pre_pyh_df.pop('Basophils')
    # Eosinophils       = pre_pyh_df.pop('Eosinophils')
    # Platelet_Count    = pre_pyh_df.pop('Platelet_Count')
    #
    # # 'gangongneng',第五类
    # AlanineAminotransferase  = pre_pyh_df.pop('AlanineAminotransferase(ALT)')
    # AlkalinePhosphatase      = pre_pyh_df.pop('AlkalinePhosphatase')
    # AspirateAminotransferase = pre_pyh_df.pop('AspirateAminotransferase(AST)')
    # Albumin                  = pre_pyh_df.pop('Albumin')
    # Bilirubin                = pre_pyh_df.pop('Bilirubin')
    #
    # # 'qita'其他
    # MCH             = pre_pyh_df.pop('MCH')
    # MCHC            = pre_pyh_df.pop('MCHC')
    # MCV             = pre_pyh_df.pop('MCV')
    # RDW             = pre_pyh_df.pop('RDW')
    # Red_Blood_Cells = pre_pyh_df.pop('Red _Blood_Cells')
    # patients_age    = pre_pyh_df.pop('patients_age')

    # pre_pyh_df.pop('SOFAscore')

    original_pre_pyh_train_inputs = pre_pyh_df[:int(len(pre_pyh_df) * 0.7)]  # (4964, 43)->(5673, 42)
    original_pre_pyh_test_inputs = pre_pyh_df[int(len(pre_pyh_df) * 0.7):]  # (2128, 43)->(1419, 42)
    print('原始训练集数据\n', original_pre_pyh_train_inputs[:10])
    print('原始测试集数据\n', original_pre_pyh_test_inputs[:10])
    original_pre_pyh_train_inputs.to_csv(os.path.join(logs, 'original_pre_pyh_train_features.csv'), index=False)
    original_pre_pyh_test_inputs.to_csv(os.path.join(logs, 'original_pre_pyh_test_features.csv'), index=False)

    standard_pre_pyh_df = stander_data(pre_pyh_df)
    standard_pre_pyh_train_inputs = standard_pre_pyh_df[:int(len(pre_pyh_df) * 0.7)].to_numpy()
    standard_pre_pyh_test_inputs = standard_pre_pyh_df[int(len(pre_pyh_df) * 0.7):].to_numpy()
    print('标准化后训练集数据\n', standard_pre_pyh_df[:int(len(pre_pyh_df) * 0.7)][:10])
    print('标准化后测试集数据\n', standard_pre_pyh_df[int(len(pre_pyh_df) * 0.7):][:10])

    return standard_pre_pyh_train_inputs, pre_pyh_train_labels, standard_pre_pyh_test_inputs, pre_pyh_test_labels

def pyh_dataloader(path:str, logs):
    pre_pyh_train_factors, pre_pyh_train_labels, pre_pyh_test_factors, \
    pre_pyh_test_labels = get_data(path, logs)

    pre_pyh_train_factors = torch.FloatTensor(pre_pyh_train_factors)  # torch.float32
    pre_pyh_test_factors = torch.FloatTensor(pre_pyh_test_factors)  # torch.float32

    return pre_pyh_train_factors, pre_pyh_train_labels, pre_pyh_test_factors, pre_pyh_test_labels

# 读取原始的全区数据
def whole_region_dataloader(path:str, batch_size):
    np.random.seed(256)
    pre_whole_region_df = pd.read_csv(path).sample(frac=1)  # 全区数据有表头，无标签
    pre_whole_region_df = stander_data(pre_whole_region_df)
    print('原始的全区数据', '\n', pre_whole_region_df[:10])
    pre_whole_region_factors = torch.FloatTensor(pre_whole_region_df.to_numpy())   # df->>numpy->tensor

    pre_whole_region_dataset = TabDataset(data=pre_whole_region_factors, target=None)
    pre_whole_region_dataset = DataLoader(pre_whole_region_dataset, batch_size=batch_size, shuffle=False)

    return pre_whole_region_dataset

def data_screening(pre_whole_region_concat_features_prelabel_results_path,
                   num_0, num_1, al1_filter=0.9, al0_filter=0.1,):
    """
    :param num_0: 筛选补充0的数量
    :param num_1: 筛选补充1的数量
    :param al1_filter: 补充1的阈值
    :param al0_filter: 补充0的阈值
    :return:
    """
    # whole_region_data_path 全区的拼接结果，无表头
    print(al0_filter, al1_filter)
    print(num_0, num_1,' num_0, num_1')
    pre_whole_region_concat_features_prelabel_results = pd.read_csv(
        pre_whole_region_concat_features_prelabel_results_path, header=None)
    print('pre_whole_region_concat_features_prelabel_results', '\n',
          pre_whole_region_concat_features_prelabel_results[:10])
    print('pre_whole_region_concat_features_prelabel_results.shape',
          pre_whole_region_concat_features_prelabel_results.shape)   # (50000, 24)

    # type( pre_whole_region_concat_features_prelabel_results):<class 'pandas.core.frame.DataFrame'>
    pre_whole_region_concat_features_prelabel_results.columns = [i for i in range(
        len(pre_whole_region_concat_features_prelabel_results.columns))]
    print('pre_whole_region_concat_features_prelabel_results', '\n',
          pre_whole_region_concat_features_prelabel_results[:10])

    # ------------------------------------------------------#
    # n：用来指定随机抽取的样本数目（行数目）或者列数目
    # replace:False表示执行无放回抽样，True表示执行有放回抽样
    # random_state:设置随机数种子,这个参数可以复现抽样结果
    # axis=:对行进行抽样，axis=1:对列进行抽样
    # ------------------------------------------------------#
    # -1 指的是最后一列，即预测为滑坡的概率, 预测概率大于阈值1的数据
    # 从筛选后的全区数据中，只返回一部分

    pre_whole_region_concat_features_prelabel_results_1 = pre_whole_region_concat_features_prelabel_results[
        (pre_whole_region_concat_features_prelabel_results.iloc[:, -1] > al1_filter)]
    pre_whole_region_concat_features_prelabel_results_0 = pre_whole_region_concat_features_prelabel_results[
        (pre_whole_region_concat_features_prelabel_results.iloc[:, -1] < al0_filter)]
    print('pre_whole_region_concat_features_prelabel_results_0.shape',
          pre_whole_region_concat_features_prelabel_results_0.shape)
    print('pre_whole_region_concat_features_prelabel_results_1.shape',
          pre_whole_region_concat_features_prelabel_results_1.shape)

    pre_whole_region_concat_features_prelabel_results_1 = pre_whole_region_concat_features_prelabel_results_1.sample(
        n=num_1, replace=True, random_state=None, axis=0)
    pre_whole_region_concat_features_prelabel_results_0 = pre_whole_region_concat_features_prelabel_results_0.sample(
        n=num_0, replace=True, random_state=None, axis=0)

    pre_whole_region_features_1 = pre_whole_region_concat_features_prelabel_results_1.iloc[:, :-3]  # 只取出前42个特性因子
    pre_whole_region_features_0 = pre_whole_region_concat_features_prelabel_results_0.iloc[:, :-3]  # 只取出前42个特性因子
    print('pre_whole_region_features_1.shape', pre_whole_region_features_1.shape)
    print('pre_whole_region_features_0.shape', pre_whole_region_features_0.shape)

    pre_whole_region_features_1.columns = [i for i in range(pre_whole_region_features_1.shape[1])]
    pre_whole_region_features_1[pre_whole_region_features_1.shape[1]] = 1   # 添加标签列，42个特性因子+1个真实标签,包含表头
    pre_whole_region_concat_features_label_1 = pre_whole_region_features_1
    print('pre_whole_region_concat_features_label_1.shape', pre_whole_region_features_1.shape)

    pre_whole_region_features_0.columns = [i for i in range(pre_whole_region_features_0.shape[1])]
    pre_whole_region_features_0[pre_whole_region_features_0.shape[1]] = 0  # 添加标签列，42个特性因子+1个真实标签,包含表头
    pre_whole_region_concat_features_label_0 = pre_whole_region_features_0
    print('pre_whole_region_concat_features_label_0.shape', pre_whole_region_features_0.shape)

    return pre_whole_region_concat_features_label_1, pre_whole_region_concat_features_label_0            #包含表头

# whole_region_data_path 全区数据的拼接结果，无表头
def data_process(pre_pyh_train_results_path,
                 pre_pyh_test_results_path,
                 pre_pyh_train_features,
                 pre_pyh_train_label,
                 pre_pyh_test_features,
                 pre_pyh_test_label,
                 pre_whole_region_concat_features_prelabel_results_path,
                 threshold=0.6,
                 pre_pyh_save_dir=None):

    # print('pre_pyh_train_inputs.shape:', pre_pyh_train_inputs.shape)  # train_inputs: (5673, 42)
    # print('pre_pyh_train_label.shape:', pre_pyh_train_label.shape)    # train_label: torch.Size((5673）])
    # print('pre_pyh_test_label.shape:', pre_pyh_test_label.shape)      # test_label: torch.Size([1419])
    # print('pre_pyh_test_inputs.shape:', pre_pyh_test_inputs.shape)    # torch.Size(1419, 42)

    # -------------------------------------------------------#
    # 只取一半是因为原始数据中，一半标签为1的样本与真实标签为0的样本各占一半
    # 使用sample()方法返回的训练集中、验证集中标签为1、0的数据各占一半
    # -------------------------------------------------------#
    num_origin_train = int(len(pre_pyh_train_label) / 2)                      # len(train_label)=10752
    num_origin_test = int(len(pre_pyh_test_label) / 2)                        # len(train_label)=2688

    pre_pyh_train_results = pd.read_csv(pre_pyh_train_results_path, header=None)       # 训练集的预测结果,不包含表头，虽有索引项，但其索引不算列
    pre_pyh_test_results = pd.read_csv(pre_pyh_test_results_path, header=None)         # 测试集的预测结果,不包含表头，虽有索引项，但其索引不算列
    # print('pre_pyh_train_results.shape',pre_pyh_train_results.shape)                 # (10752, 2)
    # print('pre_pyh_test_results.shape',pre_pyh_test_results.shape)                   # (2688, 2)

    pre_pyh_train_results = np.array(pre_pyh_train_results)                # 将pd格式转换为numpy数组格式，不包含表头和索引
    pre_pyh_test_results = np.array(pre_pyh_test_results)                  # 将pd格式转换为numpy数组格式，不包含表头和索引
    # print('pre_pyh_train_results.shape',pre_pyh_train_results.shape)     # (10752, 2)
    # print('pre_pyh_test_results.shape',pre_pyh_test_results.shape)       # (2688, 2)

    # print(pre_pyh_train_results[:10])
    # print(pre_pyh_test_results[:10])

    pre_pyh_train_concat_features_labels = np.column_stack((pre_pyh_train_features, pre_pyh_train_label))
    pre_pyh_test_concat_features_labels = np.column_stack((pre_pyh_test_features, pre_pyh_test_label))
    # print('pre_pyh_train_concat_features_labels.shape', pre_pyh_train_concat_features_labels.shape)
    # print('pre_pyh_test_concat_features_labels.shape', pre_pyh_test_concat_features_labels.shape)
    # pre_pyh_train_concat_features_labels.shape (10752, 43),42个特征+1个真实标签
    # pre_pyh_test_concat_features_labels.shape (2688, 43)，特征+真实标签

    # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
    pre_pyh_train_concat_features_labels_results = np.c_[pre_pyh_train_concat_features_labels, pre_pyh_train_results]
    pre_pyh_test_concat_features_labels_results = np.c_[pre_pyh_test_concat_features_labels, pre_pyh_test_results]
    # print('pre_pyh_train_concat_features_labels_results.shape', pre_pyh_train_concat_features_labels_results.shape)
    # pre_pyh_train_concat_features_labels_results.shape (10752, 45),特征+真实标签+预测概率值
    # pre_pyh_train_concat_features_labels_results.shape (2688, 45),特征+真实标签+预测概率值

    # numpy->>>pd
    pre_pyh_train_concat_features_labels_results = pd.DataFrame(pre_pyh_train_concat_features_labels_results)
    pre_pyh_test_concat_features_labels_results = pd.DataFrame(pre_pyh_test_concat_features_labels_results)
    # print(pre_pyh_train_concat_features_labels_results [:10])
    # print(pre_pyh_test_concat_features_labels_results[:10])
    print('筛选前训练集的形状:', pre_pyh_train_concat_features_labels_results.shape)        # 筛选前训练集的长度: (5673, 24)
    print('筛选前测试集的形状:', pre_pyh_test_concat_features_labels_results.shape)         # 筛选前测试集的长度: (1419, 24)

    # mode="a": 打开一个文件用于追加,不会清空文件内容，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
    s = open(os.path.join(pre_pyh_save_dir, "screening.txt"), mode="a")
    # 写入内容：----------------------------------------2022-06-07 21:03:08-----------------------------------------
    s.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())).center(100, "-") + "\n")

    # 偏移量
    # 添加偏度
    # 使用 DataFrame的 skew() 方法来计算所有数据属性的高斯分布偏离情况。
    # skew() 函数的结果显示了数据分布是左偏还是右偏。当数据接近0时，表示数据的偏差非常小
    print('开始自筛选'.center(100, '-'))
    # pre_pyh_train_concat_features_labels_results: 42个特性因子 + 1个真实标签 + 2个预测结果 (-1:指的是最后一列，即预测为滑坡的概率)
    pre_pyh_concat_train_test = pd.concat([pre_pyh_train_concat_features_labels_results.iloc[:, -1],
                                           pre_pyh_test_concat_features_labels_results.iloc[:, -1]])
    sk = pre_pyh_concat_train_test.skew()
    print(' pre_pyh_concat_train_test.shape', pre_pyh_concat_train_test.shape)
    sk = sk / (1 + 2 * abs(sk))
    T1 = abs(threshold * sk) ** 0.5
    T0 = 1 - T1
    print(f"阈值：{threshold}, T1:{T1} T0:{T0}")         # T1:0.3957027537424594
    s.write(f"阈值：{threshold}, T1:{T1} T0:{T0}\n")       # T0:0.6042972462575407

    # 筛选(删除)训练集
    # 只保留标签为1时，预测结果大于某一阈值的数据，以及标签为0时，预测结果低于另外一个阈值的数据
    # -1:指的是最后一列，即预测为滑坡的概率
    # -3:指的是倒数第三列，即真实标签对应的列
    # |:代表或
    train_threshold = int(len(pre_pyh_train_concat_features_labels_results)*0.8)
    print('train_threshold', train_threshold)

    pre_pyh_train_concat_features_labels_results_part1 = \
        pre_pyh_train_concat_features_labels_results.iloc[:train_threshold]
    print('pre_pyh_train_concat_features_labels_part1', pre_pyh_train_concat_features_labels_results_part1.shape)

    pre_pyh_train_concat_features_labels_results_part2 = \
        pre_pyh_train_concat_features_labels_results.iloc[train_threshold:]
    print('pre_pyh_train_concat_features_labels_part2', pre_pyh_train_concat_features_labels_results_part2.shape)

    train_cate_dict = dict(pre_pyh_train_concat_features_labels_results_part1[21].value_counts())  # 21对应的是标签列
    train_cate_array = []
    for k, v in train_cate_dict.items():
        train_cate_array.append(v)
    print(train_cate_array)
    print(dict(pre_pyh_train_concat_features_labels_results_part1[21].value_counts()))

    filtering_pyh_train_concat_features_labels_results = pre_pyh_train_concat_features_labels_results_part1[
        ((pre_pyh_train_concat_features_labels_results.iloc[:, -1] > T1) &
         (pre_pyh_train_concat_features_labels_results.iloc[:, -3] == 1) |
         (pre_pyh_train_concat_features_labels_results.iloc[:, -1] < T0) &
         (pre_pyh_train_concat_features_labels_results.iloc[:, -3] == 0))]

    # 筛选(删除)训练集
    # 只保留标签为1时，预测结果大于某一阈值的数据，以及标签为0时，预测结果低于另外一个阈值的数据
    # -1:指的是最后一列，即预测为滑坡的概率
    # -3:指的是倒数第三列，即真实标签对应的列
    test_threshold = int(len(pre_pyh_test_concat_features_labels_results) * 0.8)
    pre_pyh_test_concat_features_labels_results_part1 = \
        pre_pyh_test_concat_features_labels_results.iloc[:test_threshold]
    print('pre_pyh_test_concat_features_labels_part1', pre_pyh_test_concat_features_labels_results_part1.shape)
    pre_pyh_test_concat_features_labels_results_part2 = \
        pre_pyh_test_concat_features_labels_results.iloc[test_threshold:]
    print('pre_pyh_test_concat_features_labels_part2', pre_pyh_test_concat_features_labels_results_part2.shape)

    test_cate_dict = dict(pre_pyh_test_concat_features_labels_results_part1[21].value_counts())
    test_cate_array = []
    for k, v in test_cate_dict.items():
        test_cate_array.append(v)
    print(test_cate_array)
    print(dict(pre_pyh_train_concat_features_labels_results_part1[21].value_counts()))
    filtering_pyh_test_concat_features_labels_results = pre_pyh_test_concat_features_labels_results_part1[
        ((pre_pyh_test_concat_features_labels_results.iloc[:, -1] > T1) &
         (pre_pyh_test_concat_features_labels_results.iloc[:, -3] == 1) |
         (pre_pyh_test_concat_features_labels_results.iloc[:, -1] < T0) &
         (pre_pyh_test_concat_features_labels_results.iloc[:, -3] == 0))]

    print('剔除部分数据后训练集的形状:', filtering_pyh_train_concat_features_labels_results.shape)
    print('剔除部分数据后测试集的形状:', filtering_pyh_test_concat_features_labels_results.shape)
    s.write(f'删除后，训练集数据量：{filtering_pyh_train_concat_features_labels_results.shape}\n')
    s.write(f'删除后，测试集数据量：{filtering_pyh_test_concat_features_labels_results.shape}\n')

    # ----------------------------------------------------------------#
    # 删除训练集的数量
    # 只取一半是因为原始数据中，一半标签为1的样本与真实标签为0的样本各占一半
    # train_num_1：训练集经过自筛选后需要补充标签为1的数据的数量
    # train_num_0：训练集经过自筛选后需要补充标签为0的数据的数量
    # num_origin_train=10752/2=5376
    # 42为真实标签对应的列
    # 使用sample()方法返回的训练集中、验证集中标签为1、0的数据各占一半
    # ----------------------------------------------------------------#
    train_num_1 = train_cate_array[0] - filtering_pyh_train_concat_features_labels_results[
        filtering_pyh_train_concat_features_labels_results.iloc[:, -3] == 1].shape[0]
    train_num_0 = train_cate_array[1] - filtering_pyh_train_concat_features_labels_results[
        filtering_pyh_train_concat_features_labels_results.iloc[:, -3] == 0].shape[0]

    print("训练集需要补充标签为1的数据数量：", train_num_1, "\n训练集需要补充标签为0的数据数量：", train_num_0)
    s.write(f"训练集需要补充标签为1的数据数量：{train_num_1}, \n训练集需要补充标签为0的数据数量：{train_num_0}\n")

    # ----------------------------------------------------------------#
    # 删除训练集的数量
    # 只取一半是因为原始数据中，一半标签为1的样本与真实标签为0的样本各占一半
    # test_num_1：训练集经过自筛选后需要补充标签为1的数据的数量
    # test_num_0：训练集经过自筛选后需要补充标签为0的数据的数量
    # num_origin_test=2688/2=1344
    # 42为真实标签对应的列
    # 使用sample()方法返回的训练集中、验证集中标签为1、0的数据各占一半
    # ----------------------------------------------------------------#
    test_num_1 = test_cate_array[0] - filtering_pyh_test_concat_features_labels_results[
        filtering_pyh_test_concat_features_labels_results.iloc[:, -3] == 1].shape[0]
    test_num_0 = test_cate_array[1] - filtering_pyh_test_concat_features_labels_results[
        filtering_pyh_test_concat_features_labels_results.iloc[:, -3] == 0].shape[0]
    print("测试集需要补充标签为1的数据数量：", test_num_1, "\n测试集需要补充标签为0的数据数量：", test_num_0)
    s.write(f"测试集需要补充标签为1的数据数量：{test_num_1}, \n测试集需要补充标签为0的数据数量：{test_num_0}\n")

    # 取出特性因子+真实标签
    filtering_pyh_train_concat_features_labels = filtering_pyh_train_concat_features_labels_results.iloc[:, :-2]
    print('剔除部分数据后，filtering_pyh_train_concat_features_labels.shape', filtering_pyh_train_concat_features_labels.shape)
    filtering_pyh_test_concat_features_labels = filtering_pyh_test_concat_features_labels_results.iloc[:, :-2]
    print('剔除部分数据后，filtering_pyh_test_concat_features_labels.shape', filtering_pyh_test_concat_features_labels.shape)

    print("data_screening".center(100, "-"))
    supplement_total_1, supplement_total_0 = data_screening(pre_whole_region_concat_features_prelabel_results_path=
                                                            pre_whole_region_concat_features_prelabel_results_path,
                                                            num_0=train_num_0+test_num_0,
                                                            num_1=train_num_1+test_num_1,
                                                            al0_filter=T1, al1_filter=T0)
    print('supplement_total_0.shape', supplement_total_0.shape)
    print('supplement_total_1.shape', supplement_total_1.shape)
    # 添加 0和1

    print('train_num_0,train_num_1,test_num_0,test_num_1', train_num_0, train_num_1, test_num_0, test_num_1)
    supplement_train_addition_0 = supplement_total_0.iloc[:train_num_0]  # 训练集添加标签为0的数据
    supplement_test_addition_0 = supplement_total_0.iloc[train_num_0:test_num_0 + train_num_0, :]  # 测试集添加标签为0的数据
    supplement_train_addition_1 = supplement_total_1.iloc[:train_num_1]  # 训练集添加标签为1的数据
    supplement_test_addition_1 = supplement_total_1.iloc[train_num_1:test_num_1 + train_num_1, :]  # 测试集添加标签为1的数据

    print('supplement_train_addition_1.shape', supplement_train_addition_1.shape)
    print('supplement_train_addition_0.shape', supplement_train_addition_0.shape)
    print('supplement_test_addition_1.shape', supplement_test_addition_1.shape)
    print('supplement_test_addition_0.shape', supplement_test_addition_0.shape)

    supplement_total_train = pd.concat([supplement_train_addition_0, supplement_train_addition_1], axis=0)
    print('supplement_total_train.shape', supplement_total_train.shape)
    supplement_total_test = pd.concat([supplement_test_addition_0, supplement_test_addition_1], axis=0)
    print('supplement_total_test.shape', supplement_total_test.shape)

    # 使用pd.concat()函数，补充数据量
    print(filtering_pyh_train_concat_features_labels[:10])
    print(supplement_total_train[:10])
    post_pyh_train_total_data = pd.concat([filtering_pyh_train_concat_features_labels, supplement_total_train])
    post_pyh_test_total_data = pd.concat([filtering_pyh_test_concat_features_labels, supplement_total_test])
    print("补充部分后，训练集数据量：", post_pyh_train_total_data.shape, '\n',
          "补充部分后，测试集数据量：", post_pyh_test_total_data.shape)
    s.write(f"补充部分后，训练集数据量：{post_pyh_train_total_data.shape}\n")  # 补充后，训练集数据量： (10752, 43)
    s.write(f"补充部分后，测试集数据量：{post_pyh_test_total_data.shape}\n")  # test_inputs.shape:  (2688, 42)

    pre_pyh_train_concat_features_labels_part2 = pre_pyh_train_concat_features_labels_results_part2.iloc[:, :-2]
    print('type(pre_pyh_train_concat_features_labels_part2)', type(pre_pyh_train_concat_features_labels_part2))
    print('type(post_pyh_train_total_data)', type(post_pyh_train_total_data))
    pre_pyh_test_concat_features_labels_part2 = pre_pyh_test_concat_features_labels_results_part2.iloc[:, :-2]
    # print('pre_pyh_train_concat_features_labels_part2', '\n', pre_pyh_train_concat_features_labels_part2[:10])
    # print(pre_pyh_train_concat_features_labels_results_part1.shape)
    # print('post_pyh_train_total_data', '\n', post_pyh_train_total_data[:10])
    post_pyh_train_total_data = post_pyh_train_total_data.to_numpy()
    post_pyh_test_total_data = post_pyh_test_total_data.to_numpy()
    pre_pyh_train_concat_features_labels_part2 = pre_pyh_train_concat_features_labels_part2.to_numpy()
    pre_pyh_test_concat_features_labels_part2 = pre_pyh_test_concat_features_labels_part2.to_numpy()

    post_pyh_train_total_data = np.r_[post_pyh_train_total_data, pre_pyh_train_concat_features_labels_part2]
    post_pyh_test_total_data = np.r_[post_pyh_test_total_data, pre_pyh_test_concat_features_labels_part2]

    # post_pyh_train_total_data = pd.concat([post_pyh_train_total_data,
    #                                        pre_pyh_train_concat_features_labels_part2], axis=1)
    # post_pyh_test_total_data = pd.concat([post_pyh_test_total_data,
    #                                       pre_pyh_test_concat_features_labels_part2], axis=1)
    print("筛选、补充、合并后，最终训练集数据量：", post_pyh_train_total_data.shape, '\n',
          "筛选、补充、合并后，最终测试集数据量：", post_pyh_test_total_data.shape)
    s.write(f"筛选、补充、合并后，最终训练集数据量：{post_pyh_train_total_data.shape}\n")  # 补充后，训练集数据量： (10752, 43)
    s.write(f"筛选、补充、合并后，最终测试集数据量：{post_pyh_test_total_data.shape}\n")  # test_inputs.shape:  (2688, 42)
    s.close()


    # -------------------------------------------------------#
    #sort_values(by, axis=0, ascending=True, inplace=False, kind=‘quicksort’, na_position=‘last’)
    #axis 如果axis=0，那么by=“列名”； 如果axis=1，那么by=“行名”；
    #ascending: True则升序，可以是[True,False]，即第一字段升序，第二个降序
    #inplace: 是否用排序后的数据框替换现有的数据框 ，True,或者False
    # -------------------------------------------------------#
    # train_data.sort_values(by=12, ascending=False, inplace=True)
    # test_data.sort_values(by=[44], ascending=False, inplace=True)
    # print("补充后，训练集数据量：", train_data.shape)
    # s.write(f"补充后，训练集数据量：{train_data.shape}\n")      # 补充后，训练集数据量： (10752, 43)

    post_pyh_train_features = post_pyh_train_total_data[:, :-1]                # 只取出测试集的前42个特性因子
    post_pyh_test_features = post_pyh_test_total_data[:, :-1]

    post_pyh_train_labels = np.array(post_pyh_train_total_data[:, -1]).astype(np.uint8)  # 取出测试集的标签
    post_pyh_test_labels = np.array(post_pyh_test_total_data[:, -1]).astype(np.uint8)  # 取出测试集的标签

    print('post_pyh_train_features.shape',post_pyh_train_features.shape)
    print('post_pyh_test_features.shape', post_pyh_test_features.shape)
    print('post_pyh_train_labels.shape', post_pyh_train_labels.shape)
    print('post_pyh_test_labels.shape', post_pyh_test_labels.shape)
    print('type(post_filtering_pyh_train_labels', type(post_pyh_train_labels))
    print('post_filtering_pyh_train_labels.dtype', post_pyh_train_labels.dtype)
    return post_pyh_train_features, post_pyh_train_labels, post_pyh_test_features, post_pyh_test_labels

