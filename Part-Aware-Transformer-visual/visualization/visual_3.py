import random
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from config_vis import cfg

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import make_model

import torch

# 显示 t-SNE 可视化并叠加到图像上的函数
def show_tsne_on_image(img, tsne_results):
    # 将图像转换为浮点数并归一化到[0, 1]区间
    img = np.float32(img) / 255
    # 创建一个背景图
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    # 在图像上叠加 t-SNE 结果
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c='r', marker='x', s=50)  # 红色标记

    ax.axis('off')  # 关闭坐标轴
    return fig

# 提取模型中间层特征的方法
def extract_features(model, input_tensor):
    # 假设模型的 `forward` 方法返回一个tuple，其中第一个元素是我们需要的特征
    features = model(input_tensor)  # 这里可以根据模型架构调整
    return features

# 主函数
def main():
    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 设置预训练模型路径
    cfg.MODEL.PRETRAIN_PATH = "/22085400520/TBDE_visual/jx_vit_base_p16_224-80ecf9dd.pth"
    vit_path = "/22085400520/TBDE_visual/me/data/PAT/market/vit_base_1/vit_60.pth"
    pat_path = "/22085400520/TBDE_visual/me/data/PAT/market/vit_base/part_attention_vit_60.pth"
    data_path = "/22085400520/Part-Aware-Transformer-new/DataSets/market1501/query/"
    save_path = "/22085400520/TBDE_visual/me/self-output/out6.jpg"

    # 加载自定义部分注意力VIT模型
    model_ours = make_model(cfg, 'part_attention_vit', num_class=1)
    model_ours.load_param(pat_path)  # 加载训练好的模型参数

    model_ours.eval()  # 设置为评估模式
    model_ours.to('cuda')   # 将模型加载到GPU

    # 加载标准VIT模型
    model_vit = make_model(cfg, 'vit', num_class=1)
    model_vit.load_param(vit_path)    # 加载训练好的VIT模型参数

    model_vit.eval()   # 设置为评估模式
    model_vit.to('cuda')   # 将模型加载到GPU

    # 图像预处理，包括调整尺寸、转换为Tensor、标准化
    transform = transforms.Compose([
        transforms.Resize((256,128)),  # 调整图像大小
        transforms.ToTensor(),   # 转换为Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),   # 标准化
    ])

    input_tensor = []
    features_list = []  # 用于存储每个图像的特征

    # 准备原始的人物图像
    base_dir = data_path    # 数据路径
    img_path = os.listdir(base_dir)   # 获取图像文件列表
    random.shuffle(img_path)    # 打乱图像顺序

    # 选择要可视化的图像数量，最多30张
    length = min(30, len(img_path)) # 要可视化的图像数量
    img_list = []    # 存储图像列表
    # 逐张处理图像
    for pth in img_path[:length]:
        img = Image.open(base_dir+pth)   # 打开图像
        img = img.resize((128,256))   # 调整图像大小
        np_img = np.array(img)[:, :, ::-1] # BGR -> RGB
        input_tensor = transform(img).unsqueeze(0)     # 对图像进行预处理并加上batch维度
        input_tensor = input_tensor.cuda()   # 将图像送到GPU
        img_list.append(np_img)   # 保存原图

        # 提取特征 (假设从模型的输出中提取特征)
        features = extract_features(model_ours, input_tensor)  # 提取模型特征
        features_list.append(features.cpu().detach().numpy().flatten())  # 存储特征向量

    # 使用 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(np.array(features_list))  # 对特征进行降维

    # 可视化 t-SNE 结果
    final_img = []
    line_len = 5   # 按5张图一行

    # 按行拼接图像
    for i in range(0, len(img_list)-1, line_len):
        if i==0:
            img_line = [img_list[l] for l in range(line_len)]
            final_img = np.concatenate(img_line,axis=1)
        else:
            img_line = [img_list[i+l] for l in range(line_len)]
            x = np.concatenate(img_line,axis=1)
            final_img = np.concatenate([final_img,x],axis=0)

    # 保存最终的图像
    cv2.imwrite(save_path, final_img)

    # 叠加 t-SNE 可视化图像
    tsne_fig = show_tsne_on_image(np_img, tsne_results)
    tsne_fig.savefig("/22085400520/TBDE_visual/me/self-output/tsne_plot_2.jpg", bbox_inches='tight')

    for i, pth in enumerate(img_path[:30]):
        print(i+1, pth)   # 打印图像文件名
    print(f"save to {save_path}")   # 输出保存路径


# 程序入口
if __name__ == '__main__':
    main()   # 调用主函数
