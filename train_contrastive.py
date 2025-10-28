import argparse
import warnings
import sys
import os
import open_clip
from open_clip import get_tokenizer

def get_parser():
    """
    创建并配置命令行参数解析器，返回解析后的参数对象。

    Returns:
        argparse.Namespace: 包含解析后命令行参数的命名空间对象。
    """
    # 创建一个命令行参数解析器，设置描述信息
    parser = argparse.ArgumentParser(description="AIGCDetection @cby Training")
    # 添加模型名称参数，默认值为 'efficientnet_b0'
    parser.add_argument("--model_name", default='efficientnet_b0', help="Setting the model name", type=str)
    # 添加嵌入维度大小参数，默认值为 1024
    parser.add_argument("--embedding_size", default=1024, help="Setting the embedding_size", type=int)
    # 添加预训练层参数，默认值为 None
    parser.add_argument("--pre_layer", default=None, help="Setting the pre_layer: srm or dct", type=str)
    # 添加分类数量参数，默认值为 2
    parser.add_argument("--num_classes", default=2, help="Setting the num classes", type=int)
    # 添加是否冻结特征提取器的标志参数
    parser.add_argument('--freeze_extractor', action='store_true', help='Whether to freeze extractor?')
    # 添加模型路径参数，默认值为 None
    parser.add_argument("--model_path", default=None, help="Setting the model path", type=str)
    # 添加是否非严格加载模型的标志参数
    parser.add_argument('--no_strict', action='store_true', help='Whether to load model without strict?')
    # 添加数据集根路径参数，默认值为 '/home/law/HDD/hjs/DRCT/dataset/MSCOCO'
    parser.add_argument("--root_path", default='/home/law/HDD/hjs/DRCT/dataset/MSCOCO',
                        help="Setting the root path for dataset loader", type=str)
    # 添加伪造数据集根路径参数，注意路径中存在多余空格
    parser.add_argument("--fake_root_path", default='/home/law/HDD/hjs/DRCT/dataset/AIGC_MSCOCO',
                        help="Setting the fake root path for dataset loader", type=str)
    # 添加是否使用 DIRE 的标志参数
    parser.add_argument('--is_dire', action='store_true', help='Whether to using DIRE?')
    # 添加测试阶段后处理增强模式参数，默认值为 None
    parser.add_argument('--post_aug_mode', default=None, help='Stetting the post aug mode during test phase.')
    # 添加保存测试结果文本文件路径参数，默认值为 None
    parser.add_argument('--save_txt', default=None, help='Stetting the save_txt path.')
    # 添加伪造数据索引参数，支持多类别，用逗号分隔，默认值为 '1'
    parser.add_argument("--fake_indexes", default='1', 
                        help="Setting the fake indexes, multi class using '1,2,3,...' ", type=str)
    # 添加数据集名称参数，默认值为 'MSCOCO'
    parser.add_argument("--dataset_name", default='MSCOCO', help="Setting the dataset name", type=str)
    # 添加 GPU 设备 ID 参数，支持多 GPU，用逗号分隔，默认值为 '0'
    parser.add_argument("--device_id", default='0',
                        help="Setting the GPU id, multi gpu split by ',', such as '0,1,2,3'", type=str)
    # 添加图像输入尺寸参数，默认值为 224
    parser.add_argument("--input_size", default=224, help="Image input size", type=int)
    # 添加是否裁剪图像的标志参数
    parser.add_argument('--is_crop', action='store_true', help='Whether to crop image?')
    # 添加是否使用频谱的标志参数
    parser.add_argument('--is_spectrum', action='store_true', help='Whether to using spectrum?')
    # 添加批量大小参数，默认值为 64
    parser.add_argument("--batch_size", default=64, help="Setting the batch size", type=int)
    # 添加训练起始轮数参数，默认值为 0
    parser.add_argument("--epoch_start", default=0, help="Setting the epoch start", type=int)
    # 添加训练总轮数参数，默认值为 50
    parser.add_argument("--num_epochs", default=50, help="Setting the num epochs", type=int)
    # 添加数据加载器工作进程数量参数，默认值为 4
    parser.add_argument("--num_workers", default=4, help="Setting the num workers", type=int)
    # 添加是否使用学习率预热的标志参数
    parser.add_argument('--is_warmup', action='store_true', help='Whether to using lr warmup')
    # 添加学习率参数，默认值为 1e-3
    parser.add_argument("--lr", default=1e-3, help="Setting the learning rate", type=float)
    # 添加保存标志参数，默认值为空字符串
    parser.add_argument("--save_flag", default='', help="Setting the save flag", type=str)
    # 添加采样器模式参数，默认值为空字符串
    parser.add_argument("--sampler_mode", default='', help="Setting the sampler mode", type=str)
    # 添加是否进行测试集预测的标志参数
    parser.add_argument('--is_test', action='store_true', help='Whether to predict the test set?')
    # 添加是否使用混合精度加速的标志参数
    parser.add_argument('--is_amp', action='store_true', help='Whether to using amp autocast(使用混合精度加速)?')
    # 添加图像修复目录参数，默认值为 'full_inpainting'
    parser.add_argument("--inpainting_dir", default='full_inpainting', help="rec_image dir", type=str)
    # 添加验证或测试阈值参数，默认值为 0.5
    parser.add_argument("--threshold", default=0.5, help="Setting the valid or testing threshold.", type=float)
    # 对比学习相关参数
    # 添加对比学习的 alpha 参数，默认值为 0.3
    parser.add_argument('--alpha', default=0.3, type=float, help="Setting the alpha for contrastive learning")
    # 添加正样本间距参数，默认值为 0.0
    parser.add_argument('--pos_margin', default=0.0, type=float)
    # 添加负样本间距参数，默认值为 1.0
    parser.add_argument('--neg_margin', default=1.0, type=float)
    # 添加温度参数，默认值为 0.5
    parser.add_argument('--tau', default=0.5, type=float)
    # 添加损失函数名称参数，默认值为 'ContrastiveLoss'
    parser.add_argument('--loss_name', default='ContrastiveLoss', type=str)
    # 添加是否使用 miner 的标志参数
    parser.add_argument('--use_miner', action='store_true', help='Whether to using miner')
    # 添加内存大小参数，默认值为 None
    parser.add_argument('--memory_size', default=None, type=int)
    # 添加用于修改配置选项的剩余命令行参数
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # 解析命令行参数
    args = parser.parse_args()

    return args


# 忽略 Python 运行时产生的警告信息，避免警告信息干扰程序的正常输出
warnings.filterwarnings("ignore")

# 将当前目录的上一级目录添加到 Python 的模块搜索路径中
# 这样在导入模块时，Python 解释器会在上一级目录中查找相应的模块
# sys.path.append('..')

# 调用 get_parser 函数，解析命令行参数，并将解析结果存储在 args 变量中
args = get_parser()

# 设置 CUDA 可见的 GPU 设备 ID，通过命令行参数指定的 device_id 来确定
# 这样 PyTorch 在运行时会只使用指定的 GPU 设备
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

# 导入 PyTorch 库，用于深度学习任务，如张量操作、模型构建等
import torch
# 导入 PyTorch 的神经网络模块，包含各种神经网络层和损失函数
import torch.nn as nn
# 导入 PyTorch 的优化器模块，用于定义和使用优化算法
import torch.optim as optim
# 从 catalyst 库的 data 模块导入 BalanceClassSampler 类，用于处理数据集中类别不平衡问题
from catalyst.data import BalanceClassSampler
# 导入 time 模块，用于记录时间，如计算程序运行时长
import time
# 从 torch.utils.data 模块导入 DataLoader 类，用于批量加载数据
from torch.utils.data import DataLoader
# 从 torch.cuda.amp 模块导入 autocast 和 GradScaler 类，用于混合精度训练，减少内存使用并加速计算
from torch.cuda.amp import autocast, GradScaler
# 从 tqdm 库导入 tqdm 类，用于在训练和评估过程中显示进度条
from tqdm import tqdm
# 导入 gc 模块（垃圾回收模块），用于手动触发垃圾回收，释放不再使用的内存
import gc
# 从 sklearn.metrics 模块导入多个评估指标函数，用于评估模型性能
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, f1_score
# 导入 pytorch_warmup 库，用于学习率预热策略
import pytorch_warmup as warmup

# 从 utils.utils 模块导入自定义的 Logger、AverageMeter 和 calculate_fnr 类和函数
# Logger 用于记录训练和验证日志，AverageMeter 用于计算和更新平均值，calculate_fnr 用于计算假阴性率
from utils.utils import Logger, AverageMeter, calculate_fnr
# 从 network.models 模块导入 get_models 函数，用于获取指定的模型
from network.models import get_models
# 从 data.dataset 模块导入 AIGCDetectionDataset 类、CLASS2LABEL_MAPPING 字典和 GenImage_LIST 列表
# AIGCDetectionDataset 用于构建自定义数据集，CLASS2LABEL_MAPPING 是类别到标签的映射，GenImage_LIST 是生成图像的列表
from data.dataset import AIGCDetectionDataset, CLASS2LABEL_MAPPING, GenImage_LIST, FF_LIST
# 从 utils.losses 模块导入 LabelSmoothing、CombinedLoss 和 FocalLoss 类，用于定义损失函数
from utils.losses import LabelSmoothing, CombinedLoss, FocalLoss
# 从 data.transform 模块导入 create_train_transforms 和 create_val_transforms 函数
# 用于创建训练集和验证集的数据预处理转换操作
from data.transform import create_train_transforms, create_val_transforms
from data.transform import create_sdie_transforms, create_sdie_transforms_dup



def merge_tensor(img, label, is_train=True):
    """
    合并图像张量和标签张量，并在训练阶段对合并后的张量进行随机打乱。

    Args:
        img (list or torch.Tensor): 图像数据，可以是张量列表或单个张量。
        label (list or torch.Tensor): 标签数据，可以是张量列表或单个张量。
        is_train (bool, optional): 是否处于训练阶段，默认为 True。

    Returns:
        tuple: 合并后的图像张量和标签张量。
    """
    def shuffle_tensor(img, label):
        """
        对输入的图像张量和标签张量进行随机打乱。

        Args:
            img (torch.Tensor): 图像张量。
            label (torch.Tensor): 标签张量。

        Returns:
            tuple: 打乱后的图像张量和标签张量。
        """
        # 生成一个随机排列的索引
        indices = torch.randperm(img.size(0))# img 的形状为 (batch_size, channels, height, width)
        # 根据随机索引对图像和标签张量进行重排
        return img[indices], label[indices]
    # 检查 img 和 label 是否为列表
    if isinstance(img, list) and isinstance(label, list):
        # 将图像列表和标签列表分别合并为单个张量
        img, label = torch.cat(img, dim=0), torch.cat(label, dim=0)
        # 如果处于训练阶段，则对合并后的张量进行随机打乱
        if is_train:
            img, label = shuffle_tensor(img, label)
    return img, label


def eval_model(model, epoch, eval_loader, tokenizer, is_save=True, threshold=0.5, alpha=0.5,
               save_txt=None):
    """
    评估模型在验证集或测试集上的性能。

    Args:
        model (torch.nn.Module): 待评估的模型。
        epoch (int): 当前的训练轮数。
        eval_loader (torch.utils.data.DataLoader): 验证集或测试集的数据加载器。
        is_save (bool, optional): 是否保存评估日志，默认为 True。
        threshold (float, optional): 二分类的阈值，默认为 0.5。
        alpha (float, optional): 对比损失的权重，默认为 0.5。
        save_txt (str, optional): 保存测试结果的文本文件路径，默认为 None。

    Returns:
        float or tuple: 如果 save_txt 不为 None，返回一个包含准确率、AUC、召回率、精确率、F1 分数和 FNR 的元组；
                        否则返回平均准确率。
    """
    # 将模型设置为评估模式，关闭如 Dropout 等训练时使用的特殊层
    model.eval()
    # 初始化损失平均计算器
    losses = AverageMeter()
    # 初始化准确率平均计算器
    accuracies = AverageMeter()
    # 使用 tqdm 包装评估数据加载器，用于显示评估进度
    eval_process = tqdm(eval_loader)
    # 用于存储所有样本的真实标签
    labels = []
    # 用于存储所有样本的预测输出
    outputs = []
    # 不计算梯度，减少内存消耗并加快评估速度
    with torch.no_grad():
        for i, (img, type,label) in enumerate(eval_process):
            # 合并图像和标签张量，不进行随机打乱
            # img, label = merge_tensor(img, label, is_train=False)
            # 每处理一个批次，更新进度条显示的信息
            if i > 0 and i % 1 == 0:
                eval_process.set_description("Epoch: %d, Loss: %.4f, Acc: %.4f" %
                                             (epoch, losses.avg, accuracies.avg))
            # 将图像和标签数据移动到 GPU 上
            # img, label = img.cuda(), label.cuda()
            img = [i.cuda() for i in img] # 遍历列表，分别将每个张量移到GPU
            label = label.cuda() # 标签张量也需要移动到GPU
            # 前向传播，获取模型的预测结果和嵌入向量
            batch_size = img[0].size(0)
            # y_pred, embeddings = model(img, return_feature=True)
            y_pred, _ = model(tokenizer,img , type, None)
            # 对预测结果应用 Softmax 函数，将输出转换为概率分布
            # y_pred = nn.Softmax(dim=1)(y_pred)
            # 计算总损失，包括分类损失和对比损失
            # loss = (1 - alpha) * criterion(y_pred, label) + alpha * contrastive_loss(embeddings, label)
            loss = criterion(y_pred, label) 

            # 提取正类的预测概率
            outputs.append(1 - y_pred[:, 0])
            # 存储当前批次的真实标签
            labels.append(label)
            # 计算当前批次的准确率
            acc = (torch.max(y_pred.detach(), 1)[1] == label).sum().item() / batch_size
            # 更新损失平均值
            losses.update(loss.item(), batch_size)
            # 更新准确率平均值
            accuracies.update(acc, batch_size)

    # 将所有批次的预测输出拼接成一个张量，并转换为 NumPy 数组
    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    # 将所有批次的真实标签拼接成一个张量，并转换为 NumPy 数组
    labels = torch.cat(labels, dim=0).cpu().numpy()
    # 将标签大于 0 的值都设为 1，转换为二分类标签
    labels[labels > 0] = 1
    # 计算 AUC 分数
    # auc = roc_auc_score(labels, outputs)
    # 计算召回率
    recall = recall_score(labels, outputs > threshold)
    # 计算精确率
    precision = precision_score(labels, outputs > threshold)
    # 计算二分类准确率
    binary_acc = accuracy_score(labels, outputs > threshold)
    # 计算 F1 分数
    f1 = f1_score(labels, outputs > threshold)
    # 计算 FNR（假阴性率）
    fnr = calculate_fnr(labels, outputs > threshold)
    # 打印评估指标
    # print(f'AUC:{auc}-Recall:{recall}-Precision:{precision}-BinaryAccuracy:{binary_acc}, f1: {f1}, fnr:{fnr}')
    print(f'Recall:{recall}-Precision:{precision}-BinaryAccuracy:{binary_acc}, f1: {f1}, fnr:{fnr}')
    # 如果需要保存评估日志
    if is_save:
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg, '.4f'),
            'acc': format(accuracies.avg, '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    # 打印验证集的损失和准确率
    print("Val:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))

    # 如果指定了保存测试结果的文本文件路径
    if save_txt is not None:
        return binary_acc, recall, precision, f1, fnr

    # 否则返回平均准确率
    return accuracies.avg


def train_model(model, criterion, optimizer,tokenizer, epoch, scaler=None, alpha=0.5):
    """
    在一个训练轮次中训练模型。

    Args:
        model (torch.nn.Module): 待训练的模型。
        criterion (torch.nn.Module): 分类损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        epoch (int): 当前的训练轮数。
        scaler (torch.cuda.amp.GradScaler, optional): 混合精度训练的梯度缩放器，默认为 None。
        alpha (float, optional): 对比损失的权重，默认为 0.5。
    """
    # 将模型设置为训练模式，启用如 Dropout、BatchNorm 等在训练时使用的特殊层
    model.train()
    # 初始化损失平均计算器，用于记录训练过程中的平均损失
    losses = AverageMeter()
    # 初始化准确率平均计算器，用于记录训练过程中的平均准确率
    accuracies = AverageMeter()
    # 使用 tqdm 包装训练数据加载器，用于在训练过程中显示进度条
    training_process = tqdm(train_loader)
    # 遍历训练数据加载器中的每个批次
    for i, (input, types, label) in enumerate(training_process):
        # 合并图像和标签张量，并在训练阶段对合并后的张量进行随机打乱
        # input, label = merge_tensor(input, label, is_train=True)
        # print(input)
        # print('input type:', type(input))
        # # print('input shape:', input.shape)
        # print('input[0] shape:', input[0].shape)

        # input, label = merge_tensor(input, label, is_train=True)
        # 清空优化器中之前计算的梯度信息，避免梯度累积
        model.total_steps += 1
        # print('total steps:', model.total_steps)
        optimizer.zero_grad()
        # 获取当前优化器的学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 每处理一个批次，更新进度条显示的信息，包括当前轮次、学习率、平均损失和平均准确率
        if i > 0 and i % 1 == 0:
            training_process.set_description(
                "Epoch: %d, LR: %.8f, Loss: %.4f, Acc: %.4f" % (
                    epoch, current_lr, losses.avg, accuracies.avg))

        # 将图像和标签数据移动到 GPU 上进行计算
        # input, types, label = input.cuda(), types.cuda(), label.cuda()
        input = [i.cuda() for i in input] # 同样，遍历列表处理图像
        # types 是一个字符串列表 (['real', 'fake', ...])，不需要也不能移动到GPU
        label = label.cuda()
        # 前向传播：将输入数据 x 传入模型，得到预测结果和嵌入向量
        if scaler is None:
            # 不使用混合精度训练，直接进行前向传播
            # y_pred, embeddings = model(x, return_feature=True)
            batch_size = input[0].size(0)
            preds, preds2 = model(tokenizer,input , types, None)
            # 计算总损失，包括分类损失和对比损失
            # loss = (1-alpha) * criterion(y_pred, label) + alpha * contrastive_loss(embeddings, label)

            # loss = criterion(preds, label) + criterion(preds2, label)
            loss = criterion(preds, label) 
            # 计算当前批次的准确率
            acc = (torch.max(preds.detach(), 1)[1] == label).sum().item() / batch_size
            # 更新损失平均值
            losses.update(loss.item(), batch_size)
            # 更新准确率平均值
            accuracies.update(acc, batch_size)

            # 反向传播，计算损失函数关于模型参数的梯度
            loss.backward()
            # 根据计算得到的梯度更新模型参数
            optimizer.step()
        else:
            batch_size = input[0].size(0)
            # 使用混合精度训练，在 autocast 上下文管理器中进行前向传播，以减少内存使用和加速计算
            with autocast():
                preds, preds2 = model(tokenizer,input , types, None)
                # 计算总损失，包括分类损失和对比损失
                # loss = (1-alpha) * criterion(preds, label) + alpha * criterion(preds2, label)
                loss = criterion(preds, label) 
            # 计算当前批次的准确率
            acc = (torch.max(preds.detach(), 1)[1] == label).sum().item() / batch_size
            # 更新损失平均值
            
            losses.update(loss.item(), batch_size)
            # 更新准确率平均值
            accuracies.update(acc, batch_size)

            # 使用梯度缩放器对损失进行缩放后再反向传播，避免梯度下溢
            scaler.scale(loss).backward()
            # 使用梯度缩放器更新模型参数
            scaler.step(optimizer)
            # 更新梯度缩放器的状态，为下一次迭代做准备
            scaler.update()
        if model.total_steps in [10,500,1500,3000,5000,8000,10000,12000,18000,20000,23000,25000]: # save models at these iters 
                torch.save(model.state_dict(), os.path.join(store_name, f'iter_{model.total_steps}.pth'))    
        # 如果使用学习率预热，在预热调度器的上下文中更新学习率
        if args.is_warmup:
            with warmup_scheduler.dampening():
                scheduler.step()
    # 如果不使用学习率预热，直接更新学习率
    if not args.is_warmup:
        scheduler.step()
    # 记录训练日志，包括当前轮次、平均损失、平均准确率和学习率
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg, '.4f'),
        'acc': format(accuracies.avg, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    # 打印训练集的平均损失和平均准确率
    print("Train:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))
    # 删除损失和准确率平均计算器对象，释放内存
    
    del losses, accuracies
    # 手动触发垃圾回收，清理不再使用的内存
    gc.collect()


# 示例命令，展示如何运行脚本
# python train.py --device_id=0 --model_name=efficientnet-b0 --input_size=224 --batch_size=48 --fake_indexes=1 --is_amp --save_flag=
if __name__ == '__main__':
    # 根据每个 GPU 的批次大小和可用 GPU 数量计算总的批次大小
    batch_size = args.batch_size * torch.cuda.device_count()
    # 定义日志文件的存储路径
    # writeFile = f"./output_DR-inctro_transform-drct_dire_final/{args.dataset_name}/{args.fake_indexes.replace(',', '_')}/" \
    #             f"{args.model_name.split('/')[-1]}_{args.input_size}{args.save_flag}/logs"

    #genimage路径
    writeFile = f"./output_DR-inctro_transform-drct_dire_genimage_all/{args.dataset_name}/{args.fake_indexes.replace(',', '_')}/" \
                f"{args.model_name.split('/')[-1]}_{args.input_size}{args.save_flag}/logs"
    # 定义模型权重文件的存储路径
    print(f'使用create_side_transforms')
    store_name = writeFile.replace('/logs', '/weights')
    # 打印当前使用的 GPU 设备 ID、批次大小、GPU 数量和分类数量
    print(
        f'Using gpus:{args.device_id},batch size:{batch_size},gpu_count:{torch.cuda.device_count()},num_classes:{args.num_classes}')
    # 加载模型
    model = get_models(model_name=args.model_name, num_classes=args.num_classes,
                       embedding_size=args.embedding_size, freeze_extractor=args.freeze_extractor)
    
    with open("/home/law/.cache/clip/ViT-L-14.pt", "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
    start_epoch = 0
    # model = model.module
    model.load_state_dict(checkpoint.state_dict(), strict=False)



    # 如果指定了模型路径，则加载预训练模型
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=not args.no_strict)
        print('Model found in {}'.format(args.model_path))
    else:
        # 若未指定模型路径，则初始化一个随机权重的模型
        print('No model found, initializing random model.')
    # 如果可用 GPU 数量大于 1，使用 DataParallel 进行多 GPU 训练
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        # 若只有一个 GPU，将模型移到该 GPU 上
        model = model.cuda()
    # 定义分类损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothing(smoothing=0.05).cuda(device_id)
    # 定义对比损失函数
    contrastive_loss = CombinedLoss(loss_name=args.loss_name, embedding_size=args.embedding_size,
                                    pos_margin=args.pos_margin, neg_margin=args.neg_margin, tau=args.tau,
                                    memory_size=args.memory_size, use_miner=args.use_miner, num_classes=args.num_classes)
    # 判断是否处于训练模式
    is_train = not args.is_test
    if is_train:
        # 如果存储模型权重的目录不存在，则创建该目录
        if store_name and not os.path.exists(store_name):
            os.makedirs(store_name)
        # 初始化训练日志记录器
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])
        # 设置训练数据集加载器
        xdl = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='train',
                                   num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                #    transform=create_train_transforms(size=args.input_size, is_crop=args.is_crop)
                                      transform=create_sdie_transforms(size=args.input_size, phase='train')
                                    #   transform=create_sdie_transforms_dup(size=args.input_size, phase='train')
                                   )
        # 如果指定了采样器模式，则使用 BalanceClassSampler 进行采样，否则不使用采样器
        sampler = BalanceClassSampler(labels=xdl.get_labels(), mode=args.sampler_mode) if args.sampler_mode != '' else None  # "upsampling"
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=sampler is None, num_workers=args.num_workers, sampler=sampler)
        # 获取训练数据集的长度
        train_dataset_len = len(xdl)

        # 设置验证数据集加载器
        xdl_eval = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='val',
                                        num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                        # transform=create_val_transforms(size=args.input_size, is_crop=args.is_crop)
                                        transform=create_sdie_transforms(size=args.input_size, phase='val')
                                        # transform=create_sdie_transforms_dup(size=args.input_size, phase='val')
                                        )
        eval_loader = DataLoader(xdl_eval, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        # 获取验证数据集的长度
        eval_dataset_len = len(xdl_eval)
        # 打印训练数据集和验证数据集的长度
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)
      
        tokenizer = open_clip.get_tokenizer('ViT-L-14')


        # 定义优化器为 AdamW
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-5)
        # 定义学习率调度器
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        if not args.is_warmup:
            # 若不使用学习率预热，使用余弦退火学习率调度器
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        else:
            # 若使用学习率预热，先计算总步数，再结合线性预热调度器和余弦退火调度器
            num_steps = train_dataset_len * args.num_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        # 初始化最佳准确率，若从 0 轮开始训练，初始值为 0.5，否则先评估当前模型
        best_acc = 0.5 if args.epoch_start == 0 else eval_model(model, args.epoch_start - 1, eval_loader, is_save=False)
        # 开始训练循环
        for epoch in range(args.epoch_start, args.num_epochs):
            # 训练一个轮次xaezfvaerv
            train_model(model, criterion, optimizer,tokenizer, epoch, scaler=GradScaler() if args.is_amp else None, alpha=args.alpha)
            # 每一轮或最后一轮进行验证
            if epoch % 1 == 0 or epoch == args.num_epochs - 1:
                acc = eval_model(model, epoch, eval_loader,tokenizer, alpha=args.alpha)
                # 如果当前准确率高于最佳准确率，更新最佳准确率并保存模型
                if best_acc < acc:
                    best_acc = acc
                    save_path = '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc)
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
                
                    
                # save_path = '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc)
                # if torch.cuda.device_count() > 1:
                #         torch.save(model.module.state_dict(), save_path)
                # else:
                #         torch.save(model.state_dict(), save_path)    
            # 打印当前最佳准确率
            print(f'Current best acc:{best_acc}')
        # 保存最后一轮的模型
        last_save_path = '{}/last_acc{:.4f}.pth'.format(store_name, acc)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), last_save_path)
        else:
            torch.save(model.state_dict(), last_save_path)
    else:
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        # 记录测试开始时间
        start = time.time()
        epoch_start = 1
        num_epochs = 1
        # 修正此处拼写错误，将 args.ake_indexes 改为 args.fake_indexes
        xdl_test = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes,
                                        phase='test', num_classes=args.num_classes, is_dire=args.is_dire,
                                        post_aug_mode=args.post_aug_mode, inpainting_dir=args.inpainting_dir,
                                        # transform=create_val_transforms(size=args.input_size, is_crop=args.is_crop)
                                        transform=create_sdie_transforms(size=args.input_size, phase='test')
                                        # transform=create_sdie_transforms_dup(size=args.input_size, phase='test')
                                        )
        # 设置测试数据集加载器
        test_loader = DataLoader(xdl_test, batch_size=batch_size, shuffle=False, num_workers=4)
        # 获取测试数据集的长度
        test_dataset_len = len(xdl_test)
        # 打印测试数据集的长度
        print('test_dataset_len:', test_dataset_len)
        # 评估模型在测试集上的性能
        out_metrics = eval_model(model, epoch_start, test_loader, tokenizer, is_save=False, threshold=args.threshold, save_txt=args.save_txt)
        # 打印测试总耗时
        print('Total time:', time.time() - start)
        # 保存测试结果
        if args.save_txt is not None:
            # 创建保存测试结果文件的目录
            os.makedirs(os.path.dirname(args.save_txt), exist_ok=True)
            acc, recall, precision, f1, fnr = out_metrics
            with open(args.save_txt, 'a') as file:
                if args.dataset_name == 'GenImage':
                    class_name = GenImage_LIST[int(args.fake_indexes) - 1]
                elif args.dataset_name == 'FF++':
                    class_name = FF_LIST[int(args.fake_indexes) - 1]
                else:
                    class_name = list(CLASS2LABEL_MAPPING.keys())[int(args.fake_indexes)]
                result_str = f'model_path:{args.model_path}, post_aug_mode:{args.post_aug_mode}, class_name:{class_name}\n' \
                             f'acc:{acc:.4f},  recall:{recall:.4f}, precision:{precision:.4f}, ' \
                             f'f1:{f1:.4f}, fnr: {fnr}\n'
                file.write(result_str)
            # 打印测试结果保存路径
            print(f'The result was saved in {args.save_txt}')
