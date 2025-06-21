import argparse
import random
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import torch
import torch.nn as nn
from config_vis import cfg

from vit_rollout.vit_rollout import VITAttentionRollout
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import make_model

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# 提取模型特征
def extract_features(model, img_tensor):
    # 提取某层的特征，例如vit的最后一层
    # 这里假设我们从model_ours的某个中间层提取特征
    with torch.no_grad():
        features = model.forward_features(img_tensor)  # 修改成实际的特征提取方法
    return features.cpu().numpy()


# t-SNE可视化函数
def visualize_tsne(features, labels=None):
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # 绘制t-SNE图
    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='jet', s=50)
    else:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=50)

    plt.colorbar()
    plt.title('t-SNE visualization')
    plt.show()


# 显示注意力热力图叠加在原图上的函数
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# 主函数
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg.MODEL.PRETRAIN_PATH = '/22085400520/TBDE_visual/jx_vit_base_p16_224-80ecf9dd.pth'
    cfg.MODEL.VIT_PATH = '/22085400520/TBDE_visual/me/data/PAT/market/vit_base_1/vit_60.pth'
    cfg.MODEL.PAT_PATH = '/22085400520/TBDE_visual/me/data/PAT/market/vit_base/part_attention_vit_60.pth'
    cfg.MODEL.DATA_PATH = '/22085400520/Part-Aware-Transformer-new/DataSets/market1501/query'
    cfg.MODEL.SAVE_PATH = '/22085400520/TBDE_visual/me/self-output'

    model_ours = make_model(cfg, 'part_attention_vit', num_class=1)
    checkpoint = torch.load(cfg.MODEL.PAT_PATH, map_location=device)
    model_ours.load_state_dict(checkpoint, strict=False)
    model_ours.classifier = nn.Linear(768, 1)
    model_ours.eval()
    model_ours.to('cuda')

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    input_tensor = []
    base_dir = cfg.MODEL.DATA_PATH
    img_path = os.listdir(base_dir)
    random.shuffle(img_path)

    img_list = []
    all_features = []
    all_labels = []

    for pth in img_path[:30]:
        img = Image.open(base_dir + pth)
        img = img.resize((128, 256))
        np_img = np.array(img)[:, :, ::-1]
        input_tensor = transform(img).unsqueeze(0)
        input_tensor = input_tensor.cuda()
        img_list.append(np_img)

        # 提取特征并添加到特征列表
        features = extract_features(model_ours, input_tensor)
        all_features.append(features)

        # 如果你有标签，可以在这里添加标签
        label = 0  # 或者根据需要从文件名或其他来源获取标签
        all_labels.append(label)

    # 将特征列表转换为numpy数组
    all_features = np.vstack(all_features)

    # 进行t-SNE可视化
    visualize_tsne(all_features, all_labels)

    # 保存最终的图像
    final_img = []
    line_len = 5 if len(img_list) > 1 else 3
    for i in range(0, len(img_list) - 1, line_len):
        img_line = [img_list[l] for l in range(line_len)]
        final_img = np.concatenate(img_line, axis=1)
    cv2.imwrite(cfg.MODEL.SAVE_PATH, final_img)
    print(f"save to {cfg.MODEL.SAVE_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str,
                        help="path to save your attention visualized photo. E.g., /home/me/out.jpg")
    parser.add_argument("--data_path", type=str, help="path to your dataset. E.g., dataset/market1501/query")
    parser.add_argument("--pretrain_path", type=str,
                        help="path to your pretrained vit from imagenet or else. E.g., /home/me/cpt/")
    parser.add_argument("--vit_path", type=str, help="path to your trained vanilla vit. E.g., cpt/vit.pth")
    parser.add_argument("--pat_path", type=str, help="path to your trained PAT. E.g., cpt/pat.pth")
    args = parser.parse_args()
    main(args)
