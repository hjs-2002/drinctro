import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


# 自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt

        return loss


# https://kevinmusgrave.github.io/pytorch-metric-learning/losses/
class   CombinedLoss(torch.nn.Module):
    def __init__(self, loss_name='ContrastiveLoss', embedding_size=1024, pos_margin=0.0, neg_margin=1.0,
                 memory_size=None, use_miner=False, num_classes=2, tau=0.5):
        """
        初始化 CombinedLoss 类。

        :param loss_name: 要使用的损失函数名称，默认为 'ContrastiveLoss'。
        :param embedding_size: 嵌入向量的维度，默认为 1024。
        :param pos_margin: 正样本对的边界值，默认为 0.0。
        :param neg_margin: 负样本对的边界值，默认为 1.0。
        :param memory_size: 跨批次内存的大小，默认为 None。
        :param use_miner: 是否使用样本挖掘器，默认为 False。
        :param num_classes: 分类的类别数量，默认为 2。
        :param tau: NTXentLoss 中的温度参数，默认为 0.5。
        """
        # 调用父类的构造函数
        super(CombinedLoss, self).__init__()
        # 保存损失函数名称
        self.loss_name = loss_name
        # 保存嵌入向量的维度
        self.embedding_size = embedding_size
        # 保存正样本对的边界值
        self.pos_margin = pos_margin
        # 保存负样本对的边界值
        self.neg_margin = neg_margin
        # 保存跨批次内存的大小
        self.memory_size = memory_size
        # 保存是否使用样本挖掘器的标志
        self.use_miner = use_miner
        # 保存分类的类别数量
        self.num_classes = num_classes

        # 根据损失函数名称选择对应的损失函数
        if loss_name == 'TripletMarginLoss':
            # 初始化 TripletMarginLoss，使用平滑损失
            self.loss_fn = losses.TripletMarginLoss(smooth_loss=True)
        elif loss_name == 'ArcFaceLoss':
            # 初始化 ArcFaceLoss，指定类别数量和嵌入向量维度
            self.loss_fn = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)
        elif loss_name == 'SubCenterArcFaceLoss':
            # 初始化 SubCenterArcFaceLoss，指定类别数量和嵌入向量维度
            self.loss_fn = losses.SubCenterArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)
        elif loss_name == 'CircleLoss':
            # 初始化 CircleLoss
            self.loss_fn = losses.CircleLoss()
        elif loss_name == 'NTXentLoss':
            # 初始化 NTXentLoss，指定温度参数
            self.loss_fn = losses.NTXentLoss(temperature=tau)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        else:
            # 若未匹配到上述损失函数名称，默认使用 ContrastiveLoss
            self.loss_fn = losses.ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin)

        # 根据 use_miner 标志决定是否使用 MultiSimilarityMiner 样本挖掘器
        miner = miners.MultiSimilarityMiner() if use_miner else None
        # 若 memory_size 不为 None，使用 CrossBatchMemory 包装损失函数
        if memory_size is not None:
            self.loss_fn = losses.CrossBatchMemory(self.loss_fn, embedding_size=embedding_size, memory_size=memory_size, miner=miner)

    def forward(self, embeddings, labels):
        loss = self.loss_fn(embeddings, labels)

        return loss
