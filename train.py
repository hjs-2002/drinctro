import argparse
import warnings
import sys
import os


def get_parser():
    # 创建一个 argparse.ArgumentParser 对象，用于解析命令行参数，description 为程序的描述信息
    parser = argparse.ArgumentParser(description="AIGCDetection @cby Training")
    # 添加 --model_name 参数，默认值为 'efficientnet-b0'，用于设置模型名称
    parser.add_argument("--model_name", default='efficientnet-b0', help="Setting the model name", type=str)
    # 添加 --embedding_size 参数，默认值为 None，用于设置嵌入层的大小
    parser.add_argument("--embedding_size", default=None, help="Setting the embedding_size", type=int)
    # 添加 --num_classes 参数，默认值为 2，用于设置分类的类别数量
    parser.add_argument("--num_classes", default=2, help="Setting the num classes", type=int)
    # 添加 --freeze_extractor 参数，若在命令行中指定该参数，则将其值设为 True，用于决定是否冻结特征提取器
    parser.add_argument('--freeze_extractor', action='store_true', help='Whether to freeze extractor?')
    # 添加 --model_path 参数，默认值为 None，用于设置模型文件的路径
    parser.add_argument("--model_path", default=None, help="Setting the model path", type=str)
    # 添加 --no_strict 参数，若在命令行中指定该参数，则将其值设为 True，用于决定加载模型时是否严格匹配
    parser.add_argument('--no_strict', action='store_true', help='Whether to load model without strict?')
    # 添加 --root_path 参数，默认值为 '/disk4/chenby/dataset/MSCOCO'，用于设置数据集加载的根路径
    parser.add_argument("--root_path", default='/disk4/chenby/dataset/MSCOCO',
                        help="Setting the root path for dataset loader", type=str)
    # 添加 --fake_root_path 参数，默认值为 '/disk4/chenby/dataset/DRCT-2M'，用于设置假数据的根路径
    parser.add_argument("--fake_root_path", default='/disk4/chenby/dataset/DRCT-2M',
                        help="Setting the fake root path for dataset loader", type=str)
    # 添加 --is_dire 参数，若在命令行中指定该参数，则将其值设为 True，用于决定是否使用 DIRE
    parser.add_argument('--is_dire', action='store_true', help='Whether to using DIRE?')
    # 添加 --regex 参数，默认值为 '*.*'，用于设置数据集加载时的正则表达式
    parser.add_argument("--regex", default='*.*', help="Setting the regex for dataset loader", type=str)
    # 添加 --test_all 参数，若在命令行中指定该参数，则将其值设为 True，用于决定是否进行全量测试
    parser.add_argument('--test_all', action='store_true', help='Whether to test_all?')
    # 添加 --post_aug_mode 参数，默认值为 None，用于设置测试阶段的后处理增强模式
    parser.add_argument('--post_aug_mode', default=None, help='Stetting the post aug mode during test phase.')
    # 添加 --save_txt 参数，默认值为 None，用于设置保存测试结果的文本文件路径
    parser.add_argument('--save_txt', default=None, help='Stetting the save_txt path.')
    # 添加 --fake_indexes 参数，默认值为 '1'，用于设置假数据的索引，多类别用 '1,2,3,...' 表示
    parser.add_argument("--fake_indexes", default='1',
                        help="Setting the fake indexes, multi class using '1,2,3,...' ", type=str)
    # 添加 --dataset_name 参数，默认值为 'MSCOCO'，用于设置数据集的名称
    parser.add_argument("--dataset_name", default='MSCOCO', help="Setting the dataset name", type=str)
    # 添加 --device_id 参数，默认值为 '0'，用于设置 GPU 的编号，多 GPU 用 ',' 分隔，如 '0,1,2,3'
    parser.add_argument("--device_id", default='0',
                        help="Setting the GPU id, multi gpu split by ',', such as '0,1,2,3'", type=str)
    # 添加 --input_size 参数，默认值为 224，用于设置输入图像的尺寸
    parser.add_argument("--input_size", default=224, help="Image input size", type=int)
    # 添加 --is_crop 参数，若在命令行中指定该参数，则将其值设为 True，用于决定是否裁剪图像
    parser.add_argument('--is_crop', action='store_true', help='Whether to crop image?')
    # 添加 --batch_size 参数，默认值为 64，用于设置训练或测试的批量大小
    parser.add_argument("--batch_size", default=64, help="Setting the batch size", type=int)
    # 添加 --epoch_start 参数，默认值为 0，用于设置训练开始的轮数
    parser.add_argument("--epoch_start", default=0, help="Setting the epoch start", type=int)
    # 添加 --num_epochs 参数，默认值为 50，用于设置训练的总轮数
    parser.add_argument("--num_epochs", default=50, help="Setting the num epochs", type=int)
    # 添加 --num_workers 参数，默认值为 4，用于设置数据加载的线程数
    parser.add_argument("--num_workers", default=4, help="Setting the num workers", type=int)
    # 添加 --is_warmup 参数，若在命令行中指定该参数，则将其值设为 True，用于决定是否使用学习率预热
    parser.add_argument('--is_warmup', action='store_true', help='Whether to using lr warmup')
    # 添加 --lr 参数，默认值为 1e-3，用于设置学习率
    parser.add_argument("--lr", default=1e-3, help="Setting the learning rate", type=float)
    # 添加 --save_flag 参数，默认值为 ''，用于设置保存模型时的标识
    parser.add_argument("--save_flag", default='', help="Setting the save flag", type=str)
    # 添加 --sampler_mode 参数，默认值为 ''，用于设置采样器的模式
    parser.add_argument("--sampler_mode", default='', help="Setting the sampler mode", type=str)
    # 添加 --is_test 参数，若在命令行中指定该参数，则将其值设为 True，用于决定是否对测试集进行预测
    parser.add_argument('--is_test', action='store_true', help='Whether to predict the test set?')
    # 添加 --is_amp 参数，若在命令行中指定该参数，则将其值设为 True，用于决定是否使用混合精度加速
    parser.add_argument('--is_amp', action='store_true', help='Whether to using amp autocast(使用混合精度加速)?')
    # 添加 --inpainting_dir 参数，默认值为 'full_inpainting'，用于设置修复图像的目录
    parser.add_argument("--inpainting_dir", default='full_inpainting', help="rec_image dir", type=str)
    # 添加 --threshold 参数，默认值为 0.5，用于设置验证或测试时的阈值
    parser.add_argument("--threshold", default=0.5, help="Setting the valid or testing threshold.", type=float)
    # 添加 opts 参数，用于通过命令行修改配置选项，nargs=argparse.REMAINDER 表示接收剩余的所有参数
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # 解析命令行参数并返回一个包含所有参数的命名空间对象
    args = parser.parse_args()

    return args


# 忽略 Python 运行时产生的警告信息，避免警告信息干扰程序正常输出
warnings.filterwarnings("ignore")

# 将上级目录添加到 Python 模块搜索路径中，这样程序就能导入上级目录中的模块
sys.path.append('..')

# 调用 get_parser 函数解析命令行参数，并将解析结果存储在 args 变量中
args = get_parser()

# 设置 CUDA 可见的 GPU 设备，通过命令行参数 device_id 指定要使用的 GPU 编号
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

import torch
import torch.nn as nn
import torch.optim as optim
from catalyst.data import BalanceClassSampler
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, f1_score
import pytorch_warmup as warmup

from utils.utils import Logger, AverageMeter, Test_time_agumentation, calculate_fnr
from network.models import get_models
from data.dataset import AIGCDetectionDataset, CLASS2LABEL_MAPPING, GenImage_LIST
from data.transform import create_train_transforms, create_val_transforms


def merge_tensor(img, label, is_train=True):
    """
    合并图像和标签张量，如果处于训练模式，还会对合并后的张量进行随机打乱。

    :param img: 图像张量或图像张量列表
    :param label: 标签张量或标签张量列表
    :param is_train: 是否处于训练模式，默认为 True
    :return: 合并后的图像张量和标签张量
    """
    def shuffle_tensor(img, label):
        """
        对输入的图像和标签张量进行随机打乱。

        :param img: 图像张量
        :param label: 标签张量
        :return: 打乱后的图像张量和标签张量
        """
        # 生成一个从 0 到 img 第一维长度减 1 的随机排列索引
        indices = torch.randperm(img.size(0))
        # 根据随机索引重新排列图像和标签张量
        return img[indices], label[indices]
    # 检查 img 和 label 是否为列表类型
    if isinstance(img, list) and isinstance(label, list):
        # 将图像和标签列表分别在第 0 维上进行拼接
        img, label = torch.cat(img, dim=0), torch.cat(label, dim=0)
        # 如果处于训练模式，则对拼接后的图像和标签张量进行随机打乱
        if is_train:
            img, label = shuffle_tensor(img, label)
    return img, label

def TTA(model_, img, activation=nn.Softmax(dim=1)):
    """
    实现测试时增强（Test Time Augmentation, TTA），通过对输入图像进行多种变换并聚合模型输出，提升模型预测的准确性。

    :param model_: 用于预测的深度学习模型
    :param img: 输入的图像张量
    :param activation: 激活函数，默认为 nn.Softmax(dim=1)
    :return: 经过 TTA 处理后的平均输出
    """
    # 原始图像的预测结果，作为 TTA 的基础输出，记为 1 次预测
    outputs = activation(model_(img))
    # 初始化测试时增强工具类实例，用于对图像进行各种变换
    tta = Test_time_agumentation()
    # 水平翻转和垂直翻转图像，得到 2 个翻转后的图像，共 2 次预测
    flip_imgs = tta.tensor_flip(img)
    for flip_img in flip_imgs:
        # 将翻转后图像的预测结果累加到 outputs 中
        outputs += activation(model_(flip_img))
    # 对原始图像和第一个翻转图像分别进行旋转操作，每个图像旋转后得到 3 个新图像，共 2 * 3 = 6 次预测
    for flip_img in [img, flip_imgs[0]]:
        # 对图像进行旋转操作，得到旋转后的图像列表
        rot_flip_imgs = tta.tensor_rotation(flip_img)
        for rot_flip_img in rot_flip_imgs:
            # 将旋转后图像的预测结果累加到 outputs 中
            outputs += activation(model_(rot_flip_img))

    # 总共进行了 1 + 2 + 6 = 9 次预测，将累加的输出求平均
    outputs /= 9

    return outputs


def eval_model(model, epoch, eval_loader, is_save=True, is_tta=False, threshold=0.5, save_txt=None):
    """
    评估模型在验证集或测试集上的性能。

    :param model: 待评估的深度学习模型
    :param epoch: 当前的训练轮数
    :param eval_loader: 验证集或测试集的数据加载器
    :param is_save: 是否保存评估日志，默认为 True
    :param is_tta: 是否使用测试时增强（Test Time Augmentation），默认为 False
    :param threshold: 二分类阈值，默认为 0.5
    :param save_txt: 保存评估指标的文本文件路径，默认为 None
    :return: 如果 save_txt 不为 None，返回二元准确率、AUC、召回率、精确率、F1 分数和假阴性率；否则返回平均准确率
    """
    # 将模型设置为评估模式，关闭 Dropout 和 BatchNorm 等训练时使用的特殊层
    model.eval()
    # 初始化损失值的平均记录器
    losses = AverageMeter()
    # 初始化准确率的平均记录器
    accuracies = AverageMeter()
    # 使用 tqdm 库创建一个进度条，用于显示评估过程
    eval_process = tqdm(eval_loader)
    # 用于存储所有样本的真实标签
    labels = []
    # 用于存储所有样本的模型预测输出
    outputs = []
    # 上下文管理器，关闭梯度计算，减少内存消耗并加快推理速度
    with torch.no_grad():
        for i, (img, label) in enumerate(eval_process):
            # 合并图像和标签张量，不进行随机打乱
            img, label = merge_tensor(img, label, is_train=False)
            # 每处理一个批次，更新进度条的描述信息
            if i > 0 and i % 1 == 0:
                eval_process.set_description("Epoch: %d, Loss: %.4f, Acc: %.4f" %
                                             (epoch, losses.avg, accuracies.avg))
            # 将图像和标签张量移动到 GPU 上
            img, label = img.cuda(), label.cuda()
            if not is_tta:
                # 不使用 TTA 时，直接通过模型得到预测结果
                y_pred = model(img)
                # 对预测结果应用 Softmax 激活函数，将输出转换为概率分布
                y_pred = nn.Softmax(dim=1)(y_pred)
            else:
                # 使用 TTA 时，调用 TTA 函数得到预测结果
                y_pred = TTA(model, img, activation=nn.Softmax(dim=1))
            # 提取正类的预测概率，存储到 outputs 列表中
            outputs.append(1 - y_pred[:, 0])
            # 将当前批次的真实标签存储到 labels 列表中
            labels.append(label)
            # 计算损失值
            loss = criterion(y_pred, label)
            # 计算当前批次的准确率
            # 1. 使用 torch.max(y_pred.detach(), 1)[1] 获取预测结果中每个样本概率最大的类别索引
            #    y_pred.detach() 用于切断梯度传播，避免不必要的计算和内存占用
            #    1 表示在维度 1（即类别维度）上求最大值
            #    [1] 表示取最大值对应的索引
            # 2. 将预测的类别索引与真实标签进行比较，得到一个布尔张量
            # 3. 使用 sum().item() 计算布尔张量中 True 的数量，即预测正确的样本数
            # 4. 将预测正确的样本数除以当前批次的样本数 img.size(0)，得到当前批次的准确率
            acc = (torch.max(y_pred.detach(), 1)[1] == label).sum().item() / img.size(0)

            # 更新损失值的平均记录器
            losses.update(loss.item(), img.size(0))
            # 更新准确率的平均记录器
            accuracies.update(acc, img.size(0))

    # 将所有批次的预测输出拼接成一个张量，并转换为 numpy 数组
    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    # 将所有批次的真实标签拼接成一个张量，并转换为 numpy 数组
    labels = torch.cat(labels, dim=0).cpu().numpy()
    # 将标签大于 0 的值统一设为 1，进行二分类标签处理
    labels[labels > 0] = 1
    # 计算 AUC 指标
    auc = roc_auc_score(labels, outputs)
    # 计算召回率指标
    recall = recall_score(labels, outputs > threshold)
    # 计算精确率指标
    precision = precision_score(labels, outputs > threshold)
    # 计算二元准确率指标
    binary_acc = accuracy_score(labels, outputs > threshold)
    # 计算 F1 分数指标
    f1 = f1_score(labels, outputs > threshold)
    # 计算假阴性率指标
    fnr = calculate_fnr(labels, outputs > threshold)
    # 打印各项评估指标
    print(f'AUC:{auc}-Recall:{recall}-Precision:{precision}-BinaryAccuracy:{binary_acc}, f1: {f1}, fnr:{fnr}')
    if is_save:
        # 如果需要保存日志，记录验证阶段的各项指标
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg, '.4f'),
            'acc': format(accuracies.avg, '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    # 打印验证阶段的损失值和准确率
    print("Val:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))
    # 获取平均准确率
    acc_avg = accuracies.avg
    # 释放 outputs、labels、losses 和 accuracies 占用的内存
    del outputs, labels, losses, accuracies
    # 手动触发垃圾回收，释放内存
    gc.collect()

    if save_txt is not None:
        # 如果指定了保存路径，返回各项评估指标
        return binary_acc, auc, recall, precision, f1, fnr
    # 否则返回平均准确率
    return acc_avg


def train_model(model, criterion, optimizer, epoch, scaler=None):
    """
    在一个训练周期（epoch）内训练模型。

    :param model: 待训练的深度学习模型
    :param criterion: 损失函数，用于计算模型预测结果与真实标签之间的损失
    :param optimizer: 优化器，用于更新模型的参数
    :param epoch: 当前的训练轮数
    :param scaler: 混合精度训练的梯度缩放器，默认为 None，表示不使用混合精度训练
    """
    # 将模型设置为训练模式，启用 Dropout 和 BatchNorm 等训练时使用的特殊层
    model.train()
    # 初始化损失值的平均记录器，用于记录和计算损失的平均值
    losses = AverageMeter()
    # 初始化准确率的平均记录器，用于记录和计算准确率的平均值
    accuracies = AverageMeter()
    # 使用 tqdm 库创建一个进度条，用于显示训练过程
    training_process = tqdm(train_loader)
    # 遍历训练数据加载器中的每个批次
    for i, (x, label) in enumerate(training_process):
        # 合并图像和标签张量，并在训练模式下对合并后的张量进行随机打乱
        x, label = merge_tensor(x, label, is_train=True)
        # 清空优化器中所有参数的梯度，避免梯度累积
        optimizer.zero_grad()
        # 获取当前的学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 每处理一个批次，更新进度条的描述信息
        if i > 0 and i % 1 == 0:
            training_process.set_description(
                "Epoch: %d, LR: %.8f, Loss: %.4f, Acc: %.4f" % (
                    epoch, current_lr, losses.avg, accuracies.avg))

        # 将图像和标签张量移动到 GPU 上
        x = x.cuda()
        label = label.cuda()
        # label = Variable(torch.LongTensor(label).cuda(device_id))
        # 前向传播：将输入数据 x 传入模型，得到预测结果 y_pred
        if scaler is None:
            # 不使用混合精度训练时，直接进行前向传播
            y_pred = model(x)
            # 计算损失值
            loss = criterion(y_pred, label)
            # 计算当前批次的准确率
            acc = (torch.max(y_pred.detach(), 1)[1] == label).sum().item() / x.size(0)

            # 更新损失值的平均记录器，使用 loss.item() 减少内存泄漏风险
            losses.update(loss.item(), x.size(0))
            # 更新准确率的平均记录器
            accuracies.update(acc, x.size(0))

            # 反向传播：计算损失函数关于模型参数的梯度
            loss.backward()
            # 优化器更新模型的参数
            optimizer.step()
        else:
            # 使用混合精度训练时，在 autocast 上下文中进行前向传播
            with autocast():
                y_pred = model(x)
                # 计算损失值
                loss = criterion(y_pred, label)
            # 计算当前批次的准确率
            acc = (torch.max(y_pred.detach(), 1)[1] == label).sum().item() / x.size(0)

            # 更新损失值的平均记录器
            losses.update(loss.item(), x.size(0))
            # 更新准确率的平均记录器
            accuracies.update(acc, x.size(0))

            # 对损失值进行梯度缩放后再进行反向传播
            scaler.scale(loss).backward()
            # 优化器根据缩放后的梯度更新模型参数
            scaler.step(optimizer)
            # 更新梯度缩放器的缩放因子
            scaler.update()

        # 如果使用学习率预热，在 warmup_scheduler 上下文中更新学习率
        if args.is_warmup:
            with warmup_scheduler.dampening():
                scheduler.step()
    # 如果不使用学习率预热，直接更新学习率
    if not args.is_warmup:
        scheduler.step()
    # 记录训练阶段的各项指标
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg, '.4f'),
        'acc': format(accuracies.avg, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    # 打印训练阶段的损失值和准确率
    print("Train:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))
    # 释放 losses 和 accuracies 占用的内存
    del losses, accuracies
    # 手动触发垃圾回收，释放内存
    gc.collect()


# python train.py --device_id=0 --model_name=efficientnet-b0 --input_size=224 --batch_size=48 --fake_indexes=1 --is_amp --save_flag=
if __name__ == '__main__':
    # 计算实际使用的批量大小，乘以可用 GPU 的数量
    batch_size = args.batch_size * torch.cuda.device_count()
    # 定义日志文件的存储路径
    writeFile = f"./output/{args.dataset_name}/{args.fake_indexes.replace(',', '_')}/" \
                f"{args.model_name.split('/')[-1]}_{args.input_size}{args.save_flag}/logs"
    # 定义模型权重文件的存储路径
    store_name = writeFile.replace('/logs', '/weights')
    # 打印使用的 GPU 编号、批量大小、可用 GPU 数量和分类类别数量
    print(f'Using gpus:{args.device_id},batch size:{batch_size},gpu_count:{torch.cuda.device_count()},num_classes:{args.num_classes}')
    # 加载模型
    model = get_models(model_name=args.model_name, num_classes=args.num_classes,
                       freeze_extractor=args.freeze_extractor, embedding_size=args.embedding_size)
    # 如果指定了模型路径，则加载预训练模型
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=not args.no_strict)
        print('Model found in {}'.format(args.model_path))
    else:
        print('No model found, initializing random model.')
    # 如果可用 GPU 数量大于 1，则使用 DataParallel 进行多 GPU 训练
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        # 否则将模型移到单个 GPU 上
        model = model.cuda()
    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothing(smoothing=0.05).cuda(device_id)
    # 判断是否处于训练模式
    is_train = not args.is_test
    if is_train:
        # 如果模型权重存储路径不存在，则创建该路径
        if store_name and not os.path.exists(store_name):
            os.makedirs(store_name)
        # 初始化训练日志记录器
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])
        # 设置训练数据集
        xdl = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='train',
                                   num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                   transform=create_train_transforms(size=args.input_size, is_crop=args.is_crop)
                                   )
        # 如果指定了采样器模式，则使用 BalanceClassSampler 进行采样，否则不使用采样器
        sampler = BalanceClassSampler(labels=xdl.get_labels(), mode=args.sampler_mode) if args.sampler_mode != '' else None  # "upsampling"
        # 创建训练数据加载器
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=sampler is None, num_workers=args.num_workers, sampler=sampler)
        # 获取训练数据集的长度
        train_dataset_len = len(xdl)

        # 设置验证数据集
        xdl_eval = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='val',
                                        num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                        transform=create_val_transforms(size=args.input_size, is_crop=args.is_crop)
                                        )
        # 创建验证数据加载器
        eval_loader = DataLoader(xdl_eval, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        # 获取验证数据集的长度
        eval_dataset_len = len(xdl_eval)
        # 打印训练数据集和验证数据集的长度
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)

        # 定义优化器为 AdamW，AdamW 是 Adam 优化器的改进版本，引入了权重衰减的修正
        # model.parameters() 表示优化模型的所有参数
        # lr=args.lr 从命令行参数中获取学习率
        # weight_decay=4e-5 设置权重衰减系数，用于防止过拟合
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-5)
        # 注释掉的优化器定义，可按需使用 SGD 优化器
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # 注释掉的学习率调度器定义，StepLR 会每隔固定步数对学习率进行衰减
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        # 根据是否使用学习率预热设置学习率调度器
        if not args.is_warmup:
            # 若不使用学习率预热，使用 CosineAnnealingLR 学习率调度器
            # 该调度器会让学习率随训练轮次呈余弦函数形式衰减
            # 5 表示余弦退火的周期，即学习率在 5 个周期内完成一个完整的余弦衰减过程
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        else:
            # 若使用学习率预热，先计算总的训练步数
            # train_dataset_len 为训练数据集的长度，args.num_epochs 为训练的总轮数
            num_steps = train_dataset_len * args.num_epochs
            # 使用 CosineAnnealingLR 学习率调度器，T_max 为余弦退火的最大步数
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
            # 使用 UntunedLinearWarmup 学习率预热调度器，在训练开始阶段逐渐增加学习率
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        # 如果是从第一个 epoch 开始训练，最佳准确率初始化为 0.5，否则先评估模型获取初始最佳准确率
        best_acc = 0.5 if args.epoch_start == 0 else eval_model(model, args.epoch_start - 1, eval_loader, is_save=False)
        # 开始训练循环
        for epoch in range(args.epoch_start, args.num_epochs):
            # 调用 train_model 函数进行一个 epoch 的训练
            train_model(model, criterion, optimizer, epoch, scaler=GradScaler() if args.is_amp else None)
            # 每隔 1 个 epoch 或在最后一个 epoch 进行模型评估
            if epoch % 1 == 0 or epoch == args.num_epochs - 1:
                acc = eval_model(model, epoch, eval_loader)
                # 比较当前验证集的准确率 acc 和历史最佳准确率 best_acc
                # 若当前准确率更高，则更新最佳准确率并保存模型
                if best_acc < acc:
                    # 将当前准确率更新为最佳准确率
                    best_acc = acc
                    # 构建模型保存路径，包含存储目录、当前训练轮次和当前准确率信息
                    save_path = '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc)
                    # 判断是否使用了多 GPU 训练
                    if torch.cuda.device_count() > 1:
                        # 若使用多 GPU 训练，通过 model.module.state_dict() 获取模型参数
                        # 并将其保存到指定路径
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        # 若使用单 GPU 训练，直接通过 model.state_dict() 获取模型参数
                        # 并将其保存到指定路径
                        torch.save(model.state_dict(), save_path)
            # 打印当前最佳准确率
            print(f'Current best acc:{best_acc}')
        # 保存最后一个 epoch 的模型
        last_save_path = '{}/last_acc{:.4f}.pth'.format(store_name, acc)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), last_save_path)
        else:
            torch.save(model.state_dict(), last_save_path)
    else:
        # 记录测试开始时间
        start = time.time()
        epoch_start = 1
        num_epochs = 1
        # 设置测试数据集
        xdl_test = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes,
                                        phase='test', num_classes=args.num_classes, is_dire=args.is_dire,
                                        post_aug_mode=args.post_aug_mode, regex=args.regex, inpainting_dir=args.inpainting_dir,
                                        transform=create_val_transforms(size=args.input_size, is_crop=args.is_crop)
                                        )
        # 创建测试数据加载器
        test_loader = DataLoader(xdl_test, batch_size=batch_size, shuffle=False, num_workers=4)
        # 获取测试数据集的长度
        test_dataset_len = len(xdl_test)
        # 打印测试数据集的长度
        print('test_dataset_len:', test_dataset_len)
        # 调用 eval_model 函数进行模型测试
        out_metrics = eval_model(model, epoch_start, test_loader, is_save=False, is_tta=False,
                                 threshold=args.threshold, save_txt=args.save_txt)
        # 打印测试总耗时
        print('Total time:', time.time() - start)
        # 如果指定了保存测试结果的文本文件路径，则保存测试结果
        if args.save_txt is not None:
            os.makedirs(os.path.dirname(args.save_txt), exist_ok=True)
            acc, auc, recall, precision, f1, fnr = out_metrics
            with open(args.save_txt, 'a') as file:
                if args.dataset_name == 'GenImage':
                    class_name = GenImage_LIST[int(args.fake_indexes)-1]
                elif args.dataset_name == 'FF++':
                    # Define FF_LIST if not already imported
                    FF_LIST = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']  # Update with actual class names as needed
                    class_name = FF_LIST[int(args.fake_indexes) - 1]
                else:
                    class_name = list(CLASS2LABEL_MAPPING.keys())[int(args.fake_indexes)]
                result_str = f'model_path:{args.model_path}, post_aug_mode:{args.post_aug_mode}, class_name:{class_name}\n' \
                             f'acc:{acc:.4f}, auc:{auc:.4f}, recall:{recall:.4f}, precision:{precision:.4f}, ' \
                             f'f1:{f1:.4f}, fnr: {fnr}\n'
                file.write(result_str)
            print(f'The result was saved in {args.save_txt}')

