import argparse
import os.path
import torch.cuda
from ord_tran_nobatch_model import *
from sklearn.metrics import roc_auc_score
import copy
from nobatch_utilis_data import *
from utilis_file import *
from torch.utils.tensorboard import SummaryWriter
from thop import clever_format, profile
from torchsummary import summary
import math
from captum.attr import IntegratedGradients
# from model import CNN
# from trys import select_train
# from resultimage import drawimage
# from drawroc import *

import warnings
warnings.filterwarnings("ignore")


class Instructor(object):
    def __init__(self, args):
        super(Instructor, self).__init__()
        self.args = args
        self.device = args.device
        self.pre_best_auc_model, self.pre_best_acc_model = None, None
        self.post_best_acc_model, self.post_best_auc_model = None, None

        # 生成一个以epoch、lr参数和时间戳命名的文件夹名
        self.dir_name = get_dir_name(epoch=args.epoch, lr=args.lr)
        # 创建Logs、model文件夹，并以运行时间（年月日）+batch_size + epoch + Lr命名
        self.logs_name, self.model_name = mkdir(dir_name=self.dir_name)

        # step1 通过create_dataloaders获取训练数据和验证数据
        self.pre_train_factors, self.pre_train_labels, self.pre_test_factors, self.pre_test_labels = \
            pyh_dataloader(path=args.path, logs=self.logs_name)  # batch_size=args.batch_size
        self.pre_train_len, self.pre_test_len = len(self.pre_train_labels), len(self.pre_test_labels)
        # step2 通过cfg 定义TabTransformer
        # 网络参数

        self.pre_model = Transformer(dim=self.args.dim, depth=self.args.depth, num_heads=self.args.head,
                                     attn_drop_ratio=args.attn_drop_ratio,
                                     drop_ratio=args.drop_ratio,
                                     drop_path_ratio=args.drop_path_ratio).to(self.device)  # 模型
        # self.pre_model = CNN().to(self.device)
        # step3 定义优化器和损失函数
        self.pre_optimizer = torch.optim.Adam(self.pre_model.parameters(), lr=args.lr, weight_decay=5e-4)
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9,last_epoch=-1)
        self.pre_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.pre_optimizer,
                                                                           T_max=args.epoch + 200,
                                                                           eta_min=0, last_epoch=-1)
        self.pre_loss_fn = nn.CrossEntropyLoss().to(self.device)

        self.feature_names = ['PassengerId','Survived','Pclass','Sex','Age','SibSp',
        'Parch','Fare','family_size','Embarked_C','Embarked_Q','Embarked_S'
                              ]

        self.total_contribution = []

        self.write = SummaryWriter(log_dir=self.logs_name)

    def train(self):
        pre_best_auc, pre_best_auc_acc = torch.tensor([0.], dtype=torch.float32), torch.tensor([0.], dtype=torch.float32)
        pre_best_acc, pre_best_acc_auc = torch.tensor([0.], dtype=torch.float32), torch.tensor([0.], dtype=torch.float32)
        pre_best_acc_epoch = 0
        for i in range(self.args.epoch):
            pre_start_time = time.time()
            self.pre_model.train()

            # 修改：尝试改变每一次随机读取
            # self.pre_train_factors, self.pre_train_labels = select_train(
            #     self.pre_train_factors, self.pre_train_labels)
            # self.pre_train_len, self.pre_test_len = len(self.pre_train_labels), len(self.pre_test_labels)
            # torch.unsqueeze()input的指定位置插入一维
            pre_train_outputs = self.pre_model(torch.unsqueeze(input=self.pre_train_factors, dim=1).to(self.device))
            # pre_train_outputs = self.pre_model(self.pre_train_factors.to(self.device))

            pre_train_loss = self.pre_loss_fn(pre_train_outputs, self.pre_train_labels.to(self.device))
            pre_train_loss.backward()
            self.pre_optimizer.step()
            self.pre_optimizer.zero_grad()
            pre_train_accuracy = (pre_train_outputs.detach().cpu().argmax(axis=1) == self.pre_train_labels).sum()
            # argmax(axis=1): 返回每一行最大值的索引
            self.pre_lr_scheduler.step()  # 更新学习率
            '''
            #将标量添加到 summary
            # tag (string)：数据标识符，训练集损失
            # scalar_value (float or string/blobname)：要保存的数值
            # global_step (int)：全局步值
            '''
            self.write.add_scalar(tag='pre_train_loss', scalar_value=pre_train_loss.detach().cpu(), global_step=i)
            self.write.add_scalar(tag='pre_train_acc', scalar_value=pre_train_accuracy / self.pre_train_len,
                                  global_step=i)

            self.pre_model.eval()
            with torch.no_grad():
                pre_test_outputs = self.pre_model(torch.unsqueeze(input=self.pre_test_factors, dim=1).to(self.device))
                pre_test_loss = self.pre_loss_fn(pre_test_outputs, self.pre_test_labels.to(self.device))
                pre_test_accuracy = (pre_test_outputs.cpu().argmax(axis=1) == self.pre_test_labels).sum()
            pre_auc = roc_auc_score(self.pre_test_labels, pre_test_outputs[:, -1].cpu())

            self.write.add_scalar(tag='pre_test_loss', scalar_value=pre_test_loss.cpu(), global_step=i)
            self.write.add_scalar(tag='pre_test_acc', scalar_value=pre_test_accuracy / self.pre_test_len,
                                  global_step=i)

            if (pre_test_accuracy / self.pre_test_len) > pre_best_acc:
                self.pre_best_acc_model = copy.deepcopy(self.pre_model)
                pre_best_acc = pre_test_accuracy / self.pre_test_len
                pre_best_acc_auc = pre_auc
                torch.save(self.pre_best_acc_model.state_dict(), os.path.join(self.logs_name, 'pre_best_acc_model.pth'))
                pre_best_acc_epoch = i+1
                # print('pre_best_acc_epoch', pre_best_acc_epoch)

            if pre_auc > pre_best_auc:
                pre_best_auc = pre_auc
                pre_best_auc_acc = pre_test_accuracy / self.pre_test_len

            if (i+1) % 10 == 0:
                print('pre_Epoch: {:04d}'.format(i+1),
                      'pre_train_loss: {:.4f}'.format(pre_train_loss.detach().cpu().item()),  # .item()的目的是只取张量的数值
                      'pre_train_acc: {:.4f}'.format((pre_train_accuracy / self.pre_train_len).item()),
                      'pre_val_loss: {:.4f}'.format(pre_test_loss.cpu().item()),
                      'pre_val_acc: {:.4f}'.format((pre_test_accuracy / self.pre_test_len).item()),
                      'pre_lr: {:.4f}s'.format(self.pre_optimizer.param_groups[0]['lr']),
                      'pre_time: {:.4f}s'.format(time.time() - pre_start_time))

            # # ----------------------------------#
            # # 保存训练期间的日志
            # # pathlog = self.logs_name+"train_data.txt"
            # # 保存训练数据
            # # 拼接logdata
            # logdata = 'pre_Epoch: {:04d}'.format(i + 1) + ' '
            # logdata = logdata + 'pre_train_loss: {:.4f}'.format(pre_train_loss.detach().cpu().item()) + ' '             # .item()的目的是只取张量的数值
            # logdata = logdata + 'pre_train_acc: {:.4f}'.format((pre_train_accuracy / self.pre_train_len).item()) + ' '
            # logdata = logdata + 'pre_val_loss: {:.4f}'.format(pre_test_loss.cpu().item()) + ' '
            # logdata = logdata + 'pre_val_acc: {:.4f}'.format((pre_test_accuracy / self.pre_test_len).item()) + ' '
            # logdata = logdata + 'pre_lr: {:.4f}s'.format(self.pre_optimizer.param_groups[0]['lr']) + ' '
            # logdata = logdata + 'pre_time: {:.4f}s'.format(time.time() - pre_start_time) + '\n'
            # # print(logdata)
            # # 保存logdata
            # with open(os.path.join(self.logs_name,"traindata.txt"), mode="a") as f:
            #     f.write(logdata)
        # # 绘图
        # path = os.path.join(self.logs_name, "traindata.txt")
        # logpath = os.path.join(self.logs_name, "image_ACC.jpg")
        # logpathroc = os.path.join(self.logs_name, "image_ROC.jpg")
        # epoch = self.args.epoch
        # drawimage(path, logpath, epoch)
        # # 绘制roc曲线
        # testlabel = self.pre_test_labels
        # yucelabel = pre_test_outputs[:, -1]
        # calculate_auc(testlabel, yucelabel, logpathroc)

        # # 新建日志文件保存输出日志
        # s = open(os.path.join(self.logs_name, "pre_confusion_matrix.txt"), mode="a")
        # s.write(f"depth:{'%d' % self.args.depth}\n"
        #         f"num_heads:{'%d' % self.args.head}\n"
        #         f"attn_drop_ratio:{'%s' % self.args.attn_drop_ratio}\n"
        #         f"drop_ratio:{'%s' % self.args.drop_ratio}\n"
        #         f"drop_path_ratio:{'%s' % self.args.drop_path_ratio}\n"
        #         "-------------------------------------------------\n"
        #         f"pre_best_acc:{'%.4f' % pre_best_acc}\n"
        #         f"pre_best_acc_auc:{'%.4f' % pre_best_acc_auc}\n"
        #         f"pre_best_auc:{'%.4f' % pre_best_auc}\n"
        #         f"pre_best_auc_acc:{'%.4f' % pre_best_auc_acc}\n"
        #         f"pre_best_acc_epoch:{'%.4f' % pre_best_acc_epoch}\n")
        # s.close()

        # # 新建日志文件保存输出日志，便于记录到表格中
        # s = open(os.path.join(self.logs_name,
        #                       "pre_confusion_data.txt"), mode="a")
        # s.write(f"{'%.4f' % pre_best_acc}\n"
        #         f"{'%.4f' % pre_best_acc_auc}\n"
        #         f"{'%.4f' % pre_best_auc}\n"
        #         f"{'%.4f' % pre_best_auc_acc}\n"
        #         f"{'%d' % pre_best_acc_epoch}\n")
        # s.close()
    # -----------------------------------------------#
    # 基于best_acc_model获取数据的输出结果
    # -----------------------------------------------#

    def result(self):
        print('记录')

        filtering_model = self.pre_best_acc_model
        filtering_model.eval()

        with torch.no_grad():
            train_outputs = filtering_model(torch.unsqueeze(input=self.pre_train_factors, dim=1).to(self.device))
            train_auc = roc_auc_score(self.pre_train_labels, train_outputs[:, -1].cpu())

        pre_train_prediction_labels = np.argmax(np.array(train_outputs.cpu()), axis=1)  # 预测标签
        tpo, fpo, tno, fno = 1e-8, 1e-8, 1e-8, 1e-8
        for j in range(self.pre_train_len):
            if (self.pre_train_labels.numpy())[j] == 1:
                if pre_train_prediction_labels[j] == 1:
                    tpo = tpo + 1
                else:
                    fno = fno + 1
            else:
                if pre_train_prediction_labels[j] == 1:
                    fpo = fpo + 1
                else:
                    tno = tno + 1

        ppro = tpo / (tpo + fpo)  # 查准率
        npro = tno / (tno + fno)
        recallo = tpo / (tpo + fno)  # 查全率
        precisiono = tpo / (tpo + fpo)
        F1o = (2 * precisiono * recallo) / (precisiono + recallo)
        acco = (tpo + tno) / self.pre_train_len
        MCCo = (tpo * tno - fpo * fno) / math.sqrt((tpo + fpo) * (tpo + fno) * (tno + fpo) * (tno + fno))
        s = open(os.path.join(self.logs_name, "pre_confusion_matrix.txt"), mode="a")

        s.write('{}'.format(args.path))
        s.write('\n\ntrain\n'
                f"tp          fp          tn          fn          ppr          npr \n"
                f"{'%.4f' % tpo}   {'%.4f' % fpo}   {'%.4f' % tno}   {'%.4f' % fno}   {'%.4f' % ppro}   {'%.4f' % npro}\n"
                f"accuracy  precision   recall  F1  AUC  MCC\n"
                f"{'%.4f' % acco}   {'%.4f' % precisiono}   {'%.4f' % recallo}  {'%.4f' % F1o}   {'%.4f' % train_auc}   {'%.4f' % MCCo}")

        with torch.no_grad():
            test_outputs = filtering_model(torch.unsqueeze(input=self.pre_test_factors, dim=1).to(self.device))
            test_auc = roc_auc_score(self.pre_test_labels, test_outputs[:, -1].cpu())
            test_accuracy = (test_outputs.cpu().argmax(axis=1) == self.pre_test_labels).sum()
        pre_test_prediction_labels = np.argmax(np.array(test_outputs.cpu()), axis=1)  # 预测标签
        tp, fp, tn, fn = 1e-8, 1e-8, 1e-8, 1e-8
        for j in range(self.pre_test_len):
            if (self.pre_test_labels.numpy())[j] == 1:
                if pre_test_prediction_labels[j] == 1:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if pre_test_prediction_labels[j] == 1:
                    fp = fp + 1
                else:
                    tn = tn + 1
        ppr = tp / (tp + fp)  # 查准率 精确率   precision
        npr = tn / (tn + fn)  #

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)  # 查全率
        F1 = (2 * precision * recall) / (precision + recall)
        acc = (tp + tn) / self.pre_test_len
        MCC = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        s = open(os.path.join(self.logs_name, "pre_confusion_matrix.txt"), mode="a")

        s.write('\n\ntest\n'
                f"tp          fp          tn          fn          ppr          npr\n"
                f"{'%.4f' % tp}   {'%.4f' % fp}   {'%.4f' % tn}   {'%.4f' % fn}   {'%.4f' % ppr}   {'%.4f' % npr}\n"
                f"accuracy  precision   recall  F1  AUC  MCC\n"
                f"{'%.4f' % acc}   {'%.4f' % precision}   {'%.4f' % recall}  {'%.4f' % F1}   {'%.4f' % test_auc}   {'%.4f' % MCC}")
        s.close()

    # ---------------------------------------------------------#
    # model.summary()
    # ---------------------------------------------------------#
    def model_summary(self, aim):
        # 需要使用device来指定网络在GPU还是CPU运行
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = Transformer(dim=36, depth=self.args.depth, num_heads=self.args.head,
                        attn_drop_ratio=0.,
                        drop_ratio=0.,
                        drop_path_ratio=0.).to(device)
        summary(m, (1, 36))

        dummy_input = torch.randn(1, 1, 36).to(device)
        flops, params = profile(m.to(device), (dummy_input,), verbose=False)
        # --------------------------------------------------------#
        #   flops * 2是因为profile没有将卷积作为两个operations
        #   有些论文将卷积算乘法、加法两个operations。此时乘2
        #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
        # --------------------------------------------------------#
        flops = flops * 2
        flops, params = clever_format([flops, params], "%.3f")
        print('Total GFLOPS: %s' % (flops))
        print('Total params: %s' % (params))
        s = open(os.path.join(self.logs_name, "pre_confusion_matrix.txt"), mode="a")
        s.write('-----\n'
                f"Total GFLOPS:{'%s' % flops}\n"
                f"Total params:{'%s' % params}\n")
        s.write(aim)
        s.close()

        s = open(os.path.join(self.logs_name, "pre_confusion_data.txt"), mode="a")
        s.write('-----\n'
                f"{'%s' % flops}\n"
                f"{'%s' % params}\n")
        s.close()



def hyperParameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cpu',
                        help='device')
    parser.add_argument('--attn_drop_ratio', type=float, default=0.,
                        help='Attention中，线性层后的')

    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help='#MLP中的两次dropout')

    parser.add_argument('--drop_path_ratio', type=float, default=0.,
                        help='#Attention后的随机深度')

    parser.add_argument('--dim', type=int, default=11,
                        help='#transformer的维度')

    parser.add_argument('--depth', type=int, default=8,
                        help='#transformer的深度')

    parser.add_argument('--head', type=int, default=8,
                        help='#transformer的多头数')

    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Initial learning rate.')

    parser.add_argument('--epoch', type=int, default=1500,
                        help='Number of epochs to train.')

    parser.add_argument('--batch_size', type=int, default=200,
                        help='Number of loading.')

    parser.add_argument('--warm_up_epochs', type=int, default=10000 / 15,
                        help='Number of epochs to warmup.')

    parser.add_argument('--seed', type=int, default=256, help='Random seed.')
    parser.add_argument('--path', type=str, default='data/train_chuli.csv', help='data path.')
    opts = parser.parse_args()
    return opts


if __name__=='__main__':
    print(__file__)
    args = hyperParameters()
    print(torch.cuda.is_available())
    entity = Instructor(args)    # 执行此命令，则创建了Log、model文件夹，并以运行时间（年月日）+batch_size + epoch + Lr命名
    entity.train()
    entity.result()
    aim = 'SMOTE算法上采样数据效果'
    # entity.filtering()
    # entity.test()
    # entity.model_summary(aim=aim)



