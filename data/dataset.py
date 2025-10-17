import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import os
import torch
import glob
import json
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage.filters import gaussian_filter
try:
    from transform import create_train_transforms, create_val_transforms, create_sdie_transforms
except:
    from .transform import create_train_transforms, create_val_transforms, create_sdie_transforms

# AIGC类别映射
# 定义类别名称到标签的映射字典
# 每个键值对表示一个类别名称及其对应的标签
# 其中 'real' 表示正常图像，如 MSCOCO、ImageNet 等数据集中的图像
# 其他键为不同的 AI 生成图像模型名称，对应不同的标签
CLASS2LABEL_MAPPING = {
    'real': 0,  # 正常图像, MSCOCO, ImageNet等
    'ldm-text2im-large-256': 1,  # 'CompVis/ldm-text2im-large-256': 'Latent Diffusion',  # Latent Diffusion 基础版本
    'stable-diffusion-v1-4': 2,  # 'CompVis/stable-diffusion-v1-4': 'Stable Diffusion',  # 现实版本
    'stable-diffusion-v1-5': 3,  # 'runwayml/stable-diffusion-v1-5': 'Stable Diffusion',  # 现实版本
    'stable-diffusion-2-1': 4,
    'stable-diffusion-xl-base-1.0': 5,
    'stable-diffusion-xl-refiner-1.0': 6,
    'sd-turbo': 7,
    'sdxl-turbo': 8,
    'lcm-lora-sdv1-5': 9,
    'lcm-lora-sdxl': 10,
    'sd-controlnet-canny': 11,
    'sd21-controlnet-canny': 12,
    'controlnet-canny-sdxl-1.0': 13,
    'stable-diffusion-inpainting': 14,
    'stable-diffusion-2-inpainting': 15,
    'stable-diffusion-xl-1.0-inpainting-0.1': 16,
}
# 通过字典推导式，将 CLASS2LABEL_MAPPING 中的键值对反转
# 生成标签到类别名称的映射字典
LABEL2CLASS_MAPPING = {CLASS2LABEL_MAPPING.get(key): key for key in CLASS2LABEL_MAPPING.keys()}
# 定义一个包含生成图像相关目录名称的列表
# 列表中的每个元素代表一个生成图像的数据集目录名称
GenImage_LIST = ['stable_diffusion_v_1_4/imagenet_ai_0419_sdv4', 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5',
                 'Midjourney/imagenet_midjourney', 'ADM/imagenet_ai_0508_adm', 'wukong/imagenet_ai_0424_wukong',
                 'glide/imagenet_glide', 'VQDM/imagenet_ai_0419_vqdm', 'BigGAN/imagenet_ai_0419_biggan']

FF_LIST = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

#/home/law/HDD/serein/Dataset/data/test
#/home/law/HDD/serein/Dataset/data/train
#/home/law/HDD/serein/Dataset/data/train/Deepfakes
# 抗JPEG压缩后处理测试
def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


# 抗缩放后处理测试
def cv2_scale(img, scale):
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))

    return resized_img


# 保持长宽比resize
def resize_long_size(img, long_size=512):
    scale_percent = long_size / max(img.shape[0], img.shape[1])

    # 计算新的高度和宽度
    new_width = int(img.shape[1] * scale_percent)
    new_height = int(img.shape[0] * scale_percent)

    # 调整大小
    img_resized = cv2.resize(img, (new_width, new_height))

    return img_resized


def read_image(image_path, resize_size=None):
    try:
        image = cv2.imread(image_path)
        if resize_size is not None:
            image = resize_long_size(image, long_size=resize_size)
        # Revert from BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, True
    except:
        print(f'{image_path} read error!!!')
        return np.zeros(shape=(512, 512, 3), dtype=np.uint8), False


# 同步对应打乱两个数组
def shuffle_two_array(a, b, seed=None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(b)
    return a, b


# 把标签转换为one-hot格式
def one_hot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


# 数据划分
def split_data(image_paths, labels, val_split=0.1, test_split=0.0, phase='train', seed=2022):
    """
    根据指定的划分比例和阶段，将图像路径列表和对应的标签列表划分为训练集、验证集或测试集。

    :param image_paths: 包含所有图像文件路径的列表。
    :param labels: 与图像路径列表对应的标签列表。
    :param val_split: 验证集在总数据中的比例，默认为 0.1。
    :param test_split: 测试集在总数据中的比例，默认为 0.0。
    :param phase: 当前划分阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
    :param seed: 随机数种子，用于确保数据打乱的可重复性，默认为 2022。
    :return: 划分后的图像路径列表和对应的标签列表。
    """
    # 使用 shuffle_two_array 函数同步打乱图像路径列表和标签列表，保证图像和标签的对应关系不变
    image_paths, labels = shuffle_two_array(image_paths, labels, seed=seed)
    # 获取图像路径列表的总长度，即数据总量
    total_len = len(image_paths)
    # 判断是否需要划分测试集
    if test_split > 0:
        # 根据不同的阶段确定数据划分的起始和结束索引
        if phase == 'train':
            # 训练集的起始索引为 0，结束索引为总数据量乘以 (1 - 验证集比例 - 测试集比例)
            start_index, end_index = 0, int(total_len * (1 - val_split - test_split))
        elif phase == 'val':
            # 验证集的起始索引为总数据量乘以 (1 - 验证集比例 - 测试集比例)，结束索引为总数据量乘以 (1 - 测试集比例)
            start_index, end_index = int(total_len * (1 - val_split - test_split)), int(total_len * (1 - test_split))
        else:
            # 测试集的起始索引为总数据量乘以 (1 - 测试集比例)，结束索引为总数据量
            start_index, end_index = int(total_len * (1 - test_split)), total_len
    else:
        # 若不划分测试集，仅划分训练集和验证集
        if phase == 'train':
            # 训练集的起始索引为 0，结束索引为总数据量乘以 (1 - 验证集比例)
            start_index, end_index = 0, int(total_len * (1 - val_split))
        else:
            # 验证集的起始索引为总数据量乘以 (1 - 验证集比例)，结束索引为总数据量
            start_index, end_index = int(total_len * (1 - val_split)), total_len
    # 打印当前阶段数据划分的起始和结束索引（注释掉的代码，可按需取消注释）
    # print(f'{phase} start_index-end_index:{start_index}-{end_index}')
    # 根据计算得到的起始和结束索引，截取对应的图像路径列表和标签列表
    image_paths, labels = image_paths[start_index:end_index], labels[start_index:end_index]

    return image_paths, labels


def split_dir(image_dirs, val_split=0.1, phase='train', seed=2022):
    if phase == 'all':
        return image_dirs
    image_dirs, _ = shuffle_two_array(image_dirs, image_dirs, seed=seed)
    total_len = len(image_dirs)
    if phase == 'train':
        start_index, end_index = 0, int(total_len * (1 - val_split * 2))
    elif phase == 'val':
        start_index, end_index = int(total_len * (1 - val_split * 2)), int(total_len * (1 - val_split))
    else:
        start_index, end_index = int(total_len * (1 - val_split)), total_len
    image_dirs = image_dirs[start_index:end_index]

    return image_dirs


# 获取所有图像文件
def find_images(dir_path, extensions=['.jpg', '.png', '.jpeg', '.bmp']):
    image_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.basename(file).startswith("._"):  # skip files that start with "._"
                continue
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))

    return image_files


# Calculate the DIRE
def calculate_dire(img, sdir_img, is_success=True, input_size=224, phase='train'):
    if not is_success:
        return torch.zeros(size=(3, input_size, input_size), dtype=torch.float32)
    sdie_transforms = create_sdie_transforms(size=input_size, phase=phase)

    data = sdie_transforms(image=img, rec_image=sdir_img)
    img, sdir_img = data['image'], data['rec_image']

    # norm [0,255] -> [-1, 1]
    img = img / 127.5 - 1
    sdir_img = sdir_img / 127.5 - 1
    # absolute error
    sdie = np.abs(img - sdir_img)
    sdie_tensor = torch.from_numpy(np.transpose(np.array(sdie, dtype=np.float32), [2, 0, 1]))

    return sdie_tensor


# 根据文件路径获取图片的类别名字
def get_class_name_by_path(image_path):
    if 'GenImage' in image_path:
        class_names = GenImage_LIST
        class_name = class_names[0]
        for name in class_names[1:]:
            if f'/{name}/' in image_path:
                class_name = name
                break
    else:
        class_name = 'real'
        class_names = list(CLASS2LABEL_MAPPING.keys())
        for name in class_names:
            if f'/{name}/' in image_path:
                class_name = name
                break

    return class_name


def load_DRCT_2M(real_root_path='dataset/MSCOCO',
                 fake_root_path='dataset',
                 fake_indexes='1,2,3,4,5,6', phase='train', val_split=0.1,
                 seed=2022):
    """
    加载 DRCT-2M 数据集，根据不同阶段（训练、验证、测试）划分真实和伪造图像数据。

    :param real_root_path: 真实图像的根目录，默认为 'dataset/MSCOCO'。
    :param fake_root_path: 伪造图像的根目录，默认为 'dataset'。
    :param fake_indexes: 伪造图像类别的索引，多个索引用逗号分隔，默认为 '1,2,3,4,5,6'。
    :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
    :param val_split: 验证集的划分比例，默认为 0.1。
    :param seed: 随机数种子，用于数据划分，默认为 2022。
    :return: 包含所有图像路径的列表和对应的标签列表。
    """
    # 将传入的伪造图像类别索引字符串转换为整数列表
    fake_indexes = [int(index) for index in fake_indexes.split(',')]
    # 若当前阶段不是测试阶段（即训练或验证阶段），按照 9:1 划分训练集和验证集
    if phase != 'test':  
        # 获取训练集中真实图像的所有文件路径，并按顺序排序
        real_paths = sorted(glob.glob(f"{real_root_path}/train2017/*.*"))
        # 为每个真实图像分配标签 0
        real_labels = [0 for _ in range(len(real_paths))]
        # 根据指定的划分比例和阶段，对真实图像路径和标签进行划分
        real_paths, real_labels = split_data(real_paths, real_labels, val_split=val_split, phase=phase, seed=seed)
        # 初始化伪造图像路径和标签列表
        fake_paths = []
        fake_labels = []
        # 遍历每个伪造图像类别索引
        for i, index in enumerate(fake_indexes):
            # 获取当前伪造图像类别在训练集中的所有文件路径，并按顺序排序
            fake_paths_t = sorted(glob.glob(f"{fake_root_path}/{LABEL2CLASS_MAPPING[index]}/train2017/*.*"))
            # 为每个伪造图像分配对应的标签，从 1 开始递增
            fake_labels_t = [i + 1 for _ in range(len(fake_paths_t))]
            # 根据指定的划分比例和阶段，对当前伪造图像路径和标签进行划分
            fake_paths_t, fake_labels_t = split_data(fake_paths_t, fake_labels_t, val_split=val_split, phase=phase,
                                                     seed=seed)
            # 将当前伪造图像路径添加到总伪造图像路径列表中
            fake_paths += fake_paths_t
            # 将当前伪造图像标签添加到总伪造图像标签列表中
            fake_labels += fake_labels_t
    else:  
        # 若当前阶段为测试阶段，将所有 val2017 数据作为最终测试集
        # 获取测试集中真实图像的所有文件路径，并按顺序排序
        real_paths = sorted(glob.glob(f"{real_root_path}/val2017/*.*"))
        # 为每个真实图像分配标签 0
        real_labels = [0 for _ in range(len(real_paths))]
        # 初始化伪造图像路径和标签列表
        fake_paths = []
        fake_labels = []
        # 遍历每个伪造图像类别索引
        for i, index in enumerate(fake_indexes):
            # 获取当前伪造图像类别在测试集中的所有文件路径，并按顺序排序
            fake_paths_t = sorted(glob.glob(f"{fake_root_path}/{LABEL2CLASS_MAPPING[index]}/val2017/*.*"))
            # 为每个伪造图像分配对应的标签，从 1 开始递增
            fake_labels_t = [i + 1 for _ in range(len(fake_paths_t))]
            # 将当前伪造图像路径添加到总伪造图像路径列表中
            fake_paths += fake_paths_t
            # 将当前伪造图像标签添加到总伪造图像标签列表中
            fake_labels += fake_labels_t
    # 合并真实图像路径和伪造图像路径
    image_paths = real_paths + fake_paths
    # 合并真实图像标签和伪造图像标签
    labels = real_labels + fake_labels

    # 初始化类别计数映射字典，键为类别标签，值为该类别的图像数量
    class_count_mapping = {cls: 0 for cls in range(len(fake_indexes) + 1)}
    # 统计每个类别图像的数量
    for label in labels:
        class_count_mapping[label] += 1
    # 初始化类别名称映射字典，键为类别标签，值为类别名称 
    class_name_mapping = {0: 'real'}
    # 为每个伪造图像类别标签添加对应的类别名称
    for i, fake_index in enumerate(fake_indexes):
        class_name_mapping[i + 1] = LABEL2CLASS_MAPPING[fake_index]
    # 打印当前阶段各个类别的图像数量、总图像数量以及类别名称映射
    print(f"{phase}:{class_count_mapping}, total:{len(image_paths)}, class_name_mapping:{class_name_mapping}")

    return image_paths, labels


def load_normal_data(root_path, val_split, seed, phase='train', regex='*.*', test_all=False):
    """
    加载指定目录下的图像数据及其描述。

    :param root_path: 图像数据的根目录。
    :param val_split: 验证集的划分比例。
    :param seed: 随机数种子，用于数据划分。
    :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
    :param regex: 用于过滤图像文件的正则表达式，默认为 '*.*'，表示匹配所有文件。
    :param test_all: 是否在测试阶段加载所有数据，默认为 False。
    :return: 包含图像文件路径的列表和对应的描述列表。
    """
    # 使用 glob 模块根据指定的根目录和正则表达式获取所有匹配的图像文件路径，并按顺序排序
    images_t = sorted(glob.glob(f'{root_path}/{regex}'))
    # 若 test_all 为 False，则根据当前阶段和验证集划分比例对图像数据进行划分
    if not test_all:
        images_t, _ = split_data(images_t, images_t, val_split=val_split, phase=phase, seed=seed)

    # 为每个图像文件生成一个空字符串作为描述
    captions_t = [' ' for _ in images_t]
    # 打印当前根目录下加载的图像数量
    print(f'{root_path}: {len(images_t)}')
    return images_t, captions_t


def load_GenImage(root_path='/disk1/chenby/dataset/AIGC_data/GenImage', phase='train', seed=2023,
                  indexes='1,2,3,4,5,6,7,8', val_split=0.1):
    """
    加载 GenImage 数据集，根据指定的阶段和索引选择相应的数据。

    :param root_path: GenImage 数据集的根目录，默认为 '/disk1/chenby/dataset/AIGC_data/GenImage'。
    :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
    :param seed: 随机数种子，用于数据划分，默认为 2023。
    :param indexes: 要加载的数据集目录索引，多个索引用逗号分隔，默认为 '1,2,3,4,5,6,7,8'。
    :param val_split: 验证集的划分比例，默认为 0.1。
    :return: 包含所有图像文件路径的列表和对应的标签列表。
    """
    # 将传入的索引字符串转换为整数列表，并将每个索引减 1 以匹配列表索引从 0 开始的规则
    indexes = [int(i) - 1 for i in indexes.split(',')]
    # 获取 GenImage 数据集的目录列表
    dir_list = GenImage_LIST
    # 根据转换后的索引列表，从目录列表中选取对应的目录
    selected_dir_list = [dir_list[i] for i in indexes]
    # 初始化存储真实图像路径、真实图像标签、伪造图像路径和伪造图像标签的列表
    real_images, real_labels, fake_images, fake_labels = [], [], [], []
    # 根据当前阶段确定使用的子目录，训练或验证阶段使用 'train'，测试阶段使用 'val'
    dir_phase = 'train' if phase != 'test' else 'val'
    # 遍历选中的目录列表
    for i, selected_dir in enumerate(selected_dir_list):
        # 构建真实图像所在的根目录路径
        real_root = os.path.join(root_path, selected_dir, dir_phase, 'nature')
        # 构建伪造图像所在的根目录路径
        fake_root = os.path.join(root_path, selected_dir, dir_phase, 'ai')
        # 获取当前目录下真实图像的所有文件路径，并按顺序排序
        real_images_t = sorted(glob.glob(f'{real_root}/*.*'))
        # 获取当前目录下伪造图像的所有文件路径，并按顺序排序
        fake_images_t = sorted(glob.glob(f'{fake_root}/*.*'))
        # 若当前阶段不是测试阶段，则对真实图像和伪造图像数据进行划分
        if phase != 'test':
            real_images_t, _ = split_data(real_images_t, real_images_t, val_split, phase=phase, seed=seed)
            fake_images_t, _ = split_data(fake_images_t, fake_images_t, val_split, phase=phase, seed=seed)
        # 将当前目录下的真实图像路径添加到总真实图像路径列表中
        real_images += real_images_t
        # 为当前目录下的真实图像分配标签 0，并添加到总真实图像标签列表中
        real_labels += [0 for _ in real_images_t]
        # 将当前目录下的伪造图像路径添加到总伪造图像路径列表中
        fake_images += fake_images_t
        # 为当前目录下的伪造图像分配从 1 开始递增的标签，并添加到总伪造图像标签列表中
        fake_labels += [i + 1 for _ in fake_images_t]
        # 打印当前阶段、当前目录下真实图像数量、伪造图像数量以及目录名称
        print(f'phase:{phase}, real:{len(real_images_t)}, fake-{i+1}:{len(fake_images_t)}, selected_dir:{selected_dir}')
    # 合并真实图像路径和伪造图像路径
    total_images = real_images + fake_images
    # 合并真实图像标签和伪造图像标签
    labels = real_labels + fake_labels
    # 打印当前阶段、总的真实图像数量和总的伪造图像数量
    print(f'phase:{phase}, real:{len(real_images)}, fake:{len(fake_images)}')

    return total_images, labels


def load_data(real_root_path, fake_root_path,
              phase='train', val_split=0.1, seed=2022, ):
    """
    加载真实和伪造图像数据及其标签。

    :param real_root_path: 真实图像的根目录，多个目录用逗号分隔。
    :param fake_root_path: 伪造图像的根目录，多个目录用逗号分隔。
    :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
    :param val_split: 验证集的划分比例，默认为 0.1。
    :param seed: 随机数种子，用于数据划分，默认为 2022。
    :return: 包含所有图像路径的列表和对应的标签列表。
    """
    # 初始化存储真实图像路径和对应描述的列表
    total_real_images, total_real_captions = [], []
    # 遍历由逗号分隔的真实图像根目录
    for real_root in real_root_path.split(','):
        # 调用 load_normal_data 函数加载单个目录下的真实图像和描述
        real_images_t, real_captions_t = load_normal_data(real_root, val_split, seed, phase)
        # 将当前目录下的真实图像路径添加到总列表中
        total_real_images += list(real_images_t)
        # 将当前目录下的真实图像描述添加到总列表中
        total_real_captions += list(real_captions_t)
    # 初始化存储伪造图像路径和对应描述的列表
    total_fake_images, total_fake_captions = [], []
    # 遍历由逗号分隔的伪造图像根目录
    for fake_root in fake_root_path.split(','):
        # 调用 load_normal_data 函数加载单个目录下的伪造图像和描述
        fake_images_t, fake_captions_t = load_normal_data(fake_root, val_split, seed, phase)
        # 将当前目录下的伪造图像路径添加到总列表中
        total_fake_images += list(fake_images_t)
        # 将当前目录下的伪造图像描述添加到总列表中
        total_fake_captions += list(fake_captions_t)
    # 合并真实图像路径和伪造图像路径
    image_paths = total_real_images + total_fake_images
    # 为真实图像分配标签 0，为伪造图像分配标签 1
    labels = [0 for _ in total_real_images] + [1 for _ in total_fake_images]
    # 打印当前阶段的图像总数、真实图像数和伪造图像数
    print(f'{phase}-total:{len(image_paths)}, real:{len(total_real_images)},fake:{len(total_fake_images)}')

    return image_paths, labels


def load_pair_data(root_path, fake_root_path=None, phase='train', seed=2023, fake_indexes='1',
                   inpainting_dir='full_inpainting'):
    if fake_root_path is None:  # 推理加载代码，或者用于特征提取
        assert len(root_path.split(',')) == 2
        root_path, rec_root_path = root_path.split(',')[:2]
        image_paths = sorted(glob.glob(f"{root_path}/*.*"))
        rec_image_paths = sorted(glob.glob(f"{rec_root_path}/*.*"))
        assert len(image_paths) == len(rec_image_paths)
        total_paths = []
        for image_path, rec_image_path in zip(image_paths, rec_image_paths):
            total_paths.append((image_path, rec_image_path))
        print(f'Pair data-{phase}:{len(total_paths)}.')
        return total_paths
    assert (len(root_path.split(',')) == 2 and len(fake_root_path.split(',')) == 2) or \
           (root_path == fake_root_path and 'GenImage' in root_path)
    if 'MSCOCO' in root_path:
        phase_mapping = {'train': 'train2017', 'val': 'train2017', 'test': 'val2017'}
        real_root, real_rec_root = root_path.split(',')[:2]
        # real_root = f'{real_root}/{phase_mapping[phase]}'
        # real_rec_root = f'{real_rec_root}/{inpainting_dir}/{phase_mapping[phase]}'

        print(f'real_root:{real_root}, real_rec_root:{real_rec_root}')
        fake_root, fake_rec_root = fake_root_path.split(',')[:2]
        # fake_root = f'{fake_root}/{LABEL2CLASS_MAPPING[int(fake_indexes)]}/{phase_mapping[phase]}'
        # fake_rec_root = f'{fake_rec_root}/{LABEL2CLASS_MAPPING[int(fake_indexes)]}/{inpainting_dir}/{phase_mapping[phase]}'
        # 
        print(f'fake_root:{fake_root}, fake_rec_root:{fake_rec_root}')
    elif 'DR/GenImage' in root_path:
        phase_mapping = {'train': 'train', 'val': 'train', 'test': 'val'}
        fake_indexes = int(fake_indexes)
        assert 1 <= fake_indexes <= 8 and inpainting_dir in ['inpainting', 'inpainting2', 'inpainting_xl']
        fake_name = GenImage_LIST[fake_indexes-1]
        real_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/nature/crop'
        real_rec_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/nature/{inpainting_dir}'
        fake_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/ai/crop'
        fake_rec_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/ai/{inpainting_dir}'
        print(f'fake_name:{fake_name}')
        # print(real_root, real_rec_root, fake_root, fake_rec_root)
    else:
        real_root, real_rec_root = root_path.split(',')[:2]
        fake_root, fake_rec_root = fake_root_path.split(',')[:2]
    image_paths, labels = [], []
    # load real images
    real_image_paths = sorted(glob.glob(f"{real_root}/*.*"))
    real_image_rec_paths = sorted(glob.glob(f"{real_rec_root}/*.*"))

   
    assert len(real_image_paths) == len(real_image_rec_paths) and len(real_image_paths) > 0
    total_real = len(real_image_paths)
    if phase != 'test':
        real_image_paths, real_image_rec_paths = split_data(real_image_paths, real_image_rec_paths, phase=phase, seed=seed)
    for real_image_path, real_image_rec_path in zip(real_image_paths, real_image_rec_paths):
        image_paths.append((real_image_path, real_image_rec_path))
    # load fake images
    print(f'Real data-{phase}:{len(real_image_paths)}.')
    print(f'Real rec data-{phase}:{len(real_image_rec_paths)}.')

    fake_image_paths = sorted(glob.glob(f"{fake_root}/*.*"))
    fake_image_rec_paths = sorted(glob.glob(f"{fake_rec_root}/*.*"))
    assert len(fake_image_paths) == len(fake_image_rec_paths) and len(fake_image_paths) > 0
    total_fake = len(fake_image_paths)
    if phase != 'test':
        fake_image_paths, fake_image_rec_paths  = split_data(fake_image_paths, fake_image_rec_paths, phase=phase, seed=seed)
    for fake_image_path, fake_image_rec_path in zip(fake_image_paths, fake_image_rec_paths):
        image_paths.append((fake_image_path, fake_image_rec_path))
    print(f'Fake data-{phase}:{len(fake_image_paths)}.')
    print(f'Fake rec data-{phase}:{len(fake_image_rec_paths)}.')
    labels = [0 for _ in range(len(real_image_paths))] + [1 for _ in range(len(fake_image_paths))]
    print(f'Phase:{phase}, real:{len(real_image_paths)}, fake:{len(fake_image_paths)},'
          f'Total real:{total_real}, fake:{total_fake}')

    return image_paths, labels

# def get_all_image_paths(base_dir, subfolders, exts=['jpg', 'jpeg', 'png', 'bmp', 'tiff']):
#     image_paths = []
#     for sub in subfolders:
#         sub_dir = os.path.join(base_dir, sub)
#         for ext in exts:
#             pattern = os.path.join(sub_dir, '**', f'*.{ext}')
#             image_paths.extend(glob.glob(pattern, recursive=True))
#     return image_paths
def load_ff_images(root_path='/home/law/HDD/serein/Dataset/FF++', phase='train', seed=2023,
                  indexes='1,2,3,4', val_split=0.1):
    """
    加载 test 目录下 Deepfakes、Face2Face、FaceSwap、NeuralTextures 目录的所有伪造图片（标签为1），
    以及 original 目录下的所有真实图片（标签为0）。

    :param root_path: test 数据集的根目录
    :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'
    :param seed: 随机数种子，默认为 2023
    :param indexes: 索引字符串，默认为 '1,2,3,4,5,6,7,8'
    :param val_split: 验证集划分比例，默认为 0.1
    :return: 包含所有图像文件路径的列表和对应的标签列表
    """
    indexes = [int(i) - 1 for i in indexes.split(',')]
    # 定义包含伪造图片的目录列表
    fake_dirs = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    # 定义包含真实图片的目录名
    real_dir = 'original'
    selected_dir_list = [fake_dirs[i] for i in indexes]
    # 定义支持的图片文件扩展名列表
    exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']

    # 初始化存储伪造图片路径和对应标签的列表
    fake_images, fake_labels = [], []
    # 遍历每个伪造图片目录
    for fake_dir in selected_dir_list:
        # 构建当前伪造图片目录的完整路径
        fake_dir_path = os.path.join(root_path, phase, fake_dir)
        print(f'fake_dir_path:{fake_dir_path}')
        # 初始化当前伪造图片目录下的图片计数
        count = 0
        # 遍历每个支持的文件扩展名
        
        for ext in exts:
            # 递归查找当前伪造图片目录及其子目录下所有指定扩展名的图片
            fake_images_t = sorted(glob.glob(os.path.join(fake_dir_path, '**', f'*.{ext}'), recursive=True))
            print(f'fake_images_t:{len(fake_images_t)}')
            if phase != 'test':
                fake_images_t, _ = split_data(fake_images_t, fake_images_t, val_split, phase=phase, seed=seed)
              
            # 将找到的图片路径添加到伪造图片路径列表中
            fake_images += fake_images_t*4  # 伪造图片数量乘以4，增加数据量
            # 为找到的图片添加标签 1，并添加到伪造图片标签列表中
            fake_labels += [1] * len(fake_images)
            # 更新当前伪造图片目录下的图片计数
            count += len(fake_images_t)
        # 打印当前伪造图片目录名和该目录下的图片数量
        print("fake_images:", fake_images)
        print("fake_labels:", fake_labels)
        print(f'fake_dir:{fake_dir}, fake_count:{count}')
       
    # 打印总的伪造图片数量

    # 初始化存储真实图片路径和对应标签的列表
    real_images, real_labels = [], []
    # 构建真实图片目录的完整路径
    real_dir_path = os.path.join(root_path, phase, real_dir)
    # 初始化真实图片目录下的图片计数
    count = 0
    # 遍历每个支持的文件扩展名
    for ext in exts:
        # 递归查找真实图片目录及其子目录下所有指定扩展名的图片
        real_images_t = sorted(glob.glob(os.path.join(real_dir_path, '**', f'*.{ext}'), recursive=True))
        if phase != 'test':
            real_images_t, _ = split_data(real_images_t, real_images_t, val_split, phase=phase, seed=seed)
        # 将找到的图片路径添加到真实图片路径列表中
        real_images += real_images_t
        # 为找到的图片添加标签 0，并添加到真实图片标签列表中
        real_labels += [0] * len(real_images_t)
        # 更新真实图片目录下的图片计数
        count += len(real_images_t)
    # 打印真实图片目录名和该目录下的图片数量
    print(f'real_dir:{real_dir}, real_count:{count}')

    # 合并真实图片路径和伪造图片路径
    total_images = real_images + fake_images
    # 合并真实图片标签和伪造图片标签
    labels = real_labels + fake_labels
    # 打印总的真实图片数量、总的伪造图片数量和图片总数
    print(f'total real:{len(real_images)}, total fake:{len(fake_images)}, total:{len(total_images)}')

    return total_images, labels

class AIGCDetectionDataset(Dataset):
    def __init__(self, root_path='dataset/MSCOCO', fake_root_path='/disk4/chenby/dataset/DRCT-2M',
                 fake_indexes='1,2,3,4,5,6', phase='train', is_one_hot=False, seed=2021,
                 transform=None, use_label=True, num_classes=None, regex='*.*',
                 is_dire=False, inpainting_dir='full_inpainting', post_aug_mode=None):
        """
        初始化 AIGCDetectionDataset 类的实例。

        :param root_path: real 图像的根目录，默认为 'dataset/MSCOCO'。
        :param fake_root_path: fake 图像的根目录，默认为 '/disk4/chenby/dataset/DRCT-2M'。
        :param fake_indexes: 假图像类别的索引，多个类别用逗号分隔，默认为 '1,2,3,4,5,6'。
        :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
        :param is_one_hot: 标签是否使用 one-hot 编码，默认为 False。
        :param seed: 随机数种子，用于数据划分，默认为 2021。
        :param transform: 图像数据预处理的转换函数，默认为 None。
        :param use_label: 是否使用标签，默认为 True。
        :param num_classes: 分类的类别数量，若为 None 则根据 fake_indexes 计算，默认为 None。
        :param regex: 数据过滤的正则表达式，默认为 '*.*'。
        :param is_dire: 是否使用 DIRE（Differential Representation）训练，默认为 False。
        :param inpainting_dir: 图像修复目录，默认为 'full_inpainting'。
        :param post_aug_mode: 抗后处理测试模式，例如 'blur_1', 'jpeg_30' 等，默认为 None。
        """
        # 保存 real 图像的根目录
        self.root_path = root_path  
        # 保存当前阶段
        self.phase = phase
        # 保存标签是否使用 one-hot 编码的标志
        self.is_one_hot = is_one_hot
        # 若 num_classes 为 None，则根据 fake_indexes 中类别数量加 1 计算类别数，否则直接使用传入的 num_classes
        self.num_classes = len(fake_indexes.split(',')) + 1 if num_classes is None else num_classes
        # 保存图像数据预处理的转换函数
        self.transform = transform
        # 保存是否使用标签的标志
        self.use_label = use_label
        # 保存数据过滤的正则表达式
        self.regex = regex  
        # 保存是否使用 DIRE 进行训练的标志
        self.is_dire = is_dire  
        # 保存抗后处理测试模式，例如 [blur_1, blur_2, blur_3, blur4, jpeg_30, jpeg_40, ..., jpeg_100]
        self.post_aug_mode = post_aug_mode
        # 保存随机数种子，用于数据划分
        self.seed = seed

        if use_label:
            if self.is_dire:
                # 若使用 DIRE 训练，调用 load_pair_data 函数加载成对的图像数据及其标签
                self.image_paths, self.labels = load_pair_data(root_path, fake_root_path, phase,
                                                               fake_indexes=fake_indexes,
                                                               inpainting_dir=inpainting_dir)
            elif 'MSCOCO' in root_path and len(fake_root_path.split(',')) == 1:
                # 若根目录包含 MSCOCO 且假图像根目录为单路径，调用 load_DRCT_2M 函数加载 DRCT-2M 数据集的图像数据及其标签
                self.image_paths, self.labels = load_DRCT_2M(real_root_path=root_path,
                                                             fake_root_path=fake_root_path,
                                                             fake_indexes=fake_indexes, phase=phase, seed=seed)
            elif 'GenImage' in root_path and fake_root_path == '':
            # elif 'GenImage' in root_path and fake_root_path == '/home/law/data/GenImage':    
                # 若根目录包含 GenImage 且假图像根目录为空，调用 load_GenImage 函数加载 GenImage 数据集的图像数据及其标签
                self.image_paths, self.labels = load_GenImage(root_path=root_path, phase=phase, seed=seed,
                                                              indexes=fake_indexes)
            elif 'FF++' in root_path:
                print(f'FF++ root_path:{root_path}, phase:{phase}, seed:{seed}, fake_indexes:{fake_indexes}')
                self.image_paths,self.labels = load_ff_images(root_path=root_path, phase=phase, seed=seed,
                                                              indexes=fake_indexes)
                
             
            # else:
            #     # 其他情况，调用 load_data 函数加载通用数据的图像数据及其标签
            #     self.image_paths, self.labels = load_data(real_root_path=root_path, fake_root_path=fake_root_path,
            #                                               phase=phase, seed=seed)
            else:
                # 其他情况，调用 load_data 函数加载通用数据的图像数据及其标签
                self.image_paths, self.labels = load_pair_data(root_path, fake_root_path, phase,
                                                               fake_indexes=fake_indexes,
                                                               inpainting_dir=inpainting_dir)   

            # 若类别数为 2，将标签转换为二分类标签（大于 0 的标签设为 1）
            self.labels = [int(label > 0) for label in self.labels] if self.num_classes == 2 else self.labels
        else:
            if len(root_path.split(',')) == 2 and 'DR' in root_path:
                # 若根目录为两个路径且包含 DR，使用 DIRE 并调用 load_pair_data 函数加载成对图像数据
                self.is_dire = True
                self.image_paths = load_pair_data(root_path, phase=phase, fake_indexes=fake_indexes,
                                                  inpainting_dir=inpainting_dir)
            else:
                if self.regex == 'all':
                    # 若正则表达式为 'all'，调用 find_images 函数查找指定目录下所有图像文件
                    self.image_paths = sorted(find_images(dir_path=root_path, extensions=['.jpg', '.png', '.jpeg', '.bmp']))
                else:
                    # 其他情况，根据正则表达式使用 glob 模块获取图像文件路径
                    self.image_paths = sorted(glob.glob(f'{root_path}/{self.regex}'))[:]
            # 打印预测图像的总数和使用的正则表达式
            print(f'Total predict images:{len(self.image_paths)}, regex:{self.regex}')
        if self.phase == 'test' and self.post_aug_mode is not None:
            # 若处于测试阶段且指定了后处理模式，打印后处理模式信息
            print(f"post_aug_mode:{self.post_aug_mode}, {self.post_aug_mode.split('_')[1]}")

    def __len__(self):
        return len(self.image_paths)

    def get_labels(self):
        return list(self.labels)

    def __getitem__(self, index):
        print(f'index:{index}')
        if not self.is_dire:
            # image_path = self.image_paths[index]
            # image, is_success = read_image(image_path)
            image_path, rec_image_path = self.image_paths[index]
            image, is_success = read_image(image_path)
            rec_image, rec_is_success = read_image(rec_image_path)
            is_success = is_success and rec_is_success
            print("image shape:", image.shape)
            print("rec_image shape:", rec_image.shape)
            # image_list = list()
            # image_list.append(image)
            # image_list.append(rec_image)
            
        else:
            image_path, rec_image_path = self.image_paths[index]
            image, is_success = read_image(image_path)
            rec_image, rec_is_success = read_image(rec_image_path)
            is_success = is_success and rec_is_success
            image = calculate_dire(image, rec_image, phase=self.phase)

        # 测试后处理攻击
        if self.phase == 'test' and self.post_aug_mode is not None:
            if 'jpeg' in self.post_aug_mode:
                compress_val = int(self.post_aug_mode.split('_')[1])
                image = cv2_jpg(image, compress_val)
            elif 'scale' in self.post_aug_mode:
                scale = float(self.post_aug_mode.split('_')[1])
                image = cv2_scale(image, scale)

        label = 0  # default label
        if self.use_label:
            label = self.labels[index] if is_success else 0
            if label == 0:
                type = 'real'
            else :
                type = 'fake'

        if self.transform is not None and not self.is_dire:
            try:
                if isinstance(self.transform, torchvision.transforms.transforms.Compose):
                    image = self.transform(Image.fromarray(image))
                    rec_image = self.transform(Image.fromarray(rec_image))
                    print("transform success!!!")
                    print("image shape:", image.shape)
                else:
                    data = self.transform(image=image)
                    image = data["image"]
                    rec_data = self.transform(image=rec_image) # <--- 新增
                    rec_image = rec_data["image"] # <--- 新增
            except:
                print("transform error!!!")
                image = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
                if isinstance(self.transform, torchvision.transforms.transforms.Compose):
                    image = self.transform(Image.fromarray(image))
                    rec_image = self.transform(Image.fromarray(rec_image)) # <--- 新增
                else:
                    data = self.transform(image=image)
                    image = data["image"]
                    rec_data = self.transform(image=rec_image) # <--- 新增
                    rec_image = rec_data["image"] # <--- 新增
                label = 0
        image_list = [image, rec_image]
        if not self.use_label:
            return image, image_path.replace(f"{self.root_path}", '')  # os.path.basename(image_path)

        if self.is_one_hot:
            label = one_hot(self.num_classes, label)

        return image_list, type, label




