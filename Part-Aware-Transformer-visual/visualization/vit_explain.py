import argparse
import random
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

from config_vis import cfg

from vit_rollout.vit_rollout import VITAttentionRollout
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import make_model

import torch

# 显示注意力热力图叠加在原图上的函数
def show_mask_on_image(img, mask):
    # 将图像转换为浮点数并归一化到[0, 1]区间
    img = np.float32(img) / 255
    # 将mask应用到热力图，使用Jet颜色映射
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # 将原图和热力图叠加
    cam = heatmap + np.float32(img)
    # 将结果归一化
    cam = cam / np.max(cam)
    # 将结果转为[0, 255]区间并返回
    return np.uint8(255 * cam)

# 主函数
def main(args):
    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    # 设置预训练模型路径
    cfg.MODEL.PRETRAIN_PATH = args.pretrain_path

    # load part_attention_vit  # 加载自定义的部分注意力VIT模型
    model_ours = make_model(cfg, 'part_attention_vit', num_class=1)
    model_ours.load_param(args.pat_path)  # 加载训练好的模型参数
    # model_ours.load_state_dict(torch.load(cfg.MODEL.PAT_PATH, map_location=device), strict=False)
    model_ours.eval()  # 设置为评估模式
    model_ours.to('cuda')   # 将模型加载到GPU

    # load vanilla vit   # 加载标准的VIT模型
    model_vit = make_model(cfg, 'vit', num_class=1)
    model_vit.load_param(args.vit_path)    # 加载训练好的VIT模型参数
    # model_vit.load_state_dict(torch.load(cfg.MODEL.VIT_PATH, map_location=device), strict=False)
    model_vit.eval()   # 设置为评估模式
    model_vit.to('cuda')   # 将模型加载到GPU

    # 图像预处理，包括调整尺寸、转换为Tensor、标准化
    transform = transforms.Compose([
        transforms.Resize((256,128)),  # 调整图像大小
        transforms.ToTensor(),   # 转换为Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),   # 标准化
    ])

    input_tensor = []

    # Prepare the original person photos   # 准备原始的人物图像
    base_dir = args.data_path    # 数据路径
    img_path = os.listdir(base_dir)   # 获取图像文件列表
    random.shuffle(img_path)    # 打乱图像顺序

    # 选择要可视化的图像数量，最多30张
    length = min(30, len(img_path)) # how many photos to visualize
    img_list = []    # 存储图像列表
    # 逐张处理图像
    for pth in img_path[:length]:
        img = Image.open(base_dir+pth)   # 打开图像
        img = img.resize((128,256))   # 调整图像大小
        np_img = np.array(img)[:, :, ::-1] # BGR -> RGB
        input_tensor = transform(img).unsqueeze(0)     # 对图像进行预处理并加上batch维度
        input_tensor = input_tensor.cuda()   # 将图像送到GPU
        img_list.append(np_img)   # 保存原图

        local_flag = False  # 标记是否有生成的mask

        # attention rollout  # 执行注意力展开
        for model in [model_ours]:
            attention_rollout = VITAttentionRollout(model, head_fusion='mean', discard_ratio=0.5) # modify head_fusion type and discard_ratio for better outputs # 创建注意力展开对象

            masks = attention_rollout(input_tensor)   # 获取注意力热力图

            # 如果masks是列表，则遍历并保存每一个mask
            if isinstance(masks, list):
                for msk in masks:
                    msk = cv2.resize(msk, (np_img.shape[1], np_img.shape[0]))  # 调整mask大小
                    img_list.append(show_mask_on_image(np_img, msk))   # 将mask叠加到图像并保存
                    local_flag = True   # 保存处理后的图像
            else:
                masks = cv2.resize(masks, (np_img.shape[1], np_img.shape[0]))   # 调整mask大小
                out_img = show_mask_on_image(np_img, masks)    # 将mask叠加到图像
                img_list.append(out_img)    # 保存处理后的图像

    # 将处理后的图像按行合并
    final_img = []
    line_len = 5 if local_flag else 3   # 如果有生成mask，按5张图一行，否则按3张图一行

    # concate output images in a column  # 按行拼接图像
    for i in range(0, len(img_list)-1, line_len):
        if i==0:
            img_line = [img_list[l] for l in range(line_len)]
            final_img = np.concatenate(img_line,axis=1)
        else:
            img_line = [img_list[i+l] for l in range(line_len)]
            x = np.concatenate(img_line,axis=1)
            final_img = np.concatenate([final_img,x],axis=0)

    # 保存最终的图像
    cv2.imwrite(args.save_path, final_img)
    for i, pth in enumerate(img_path[:30]):
        print(i+1, pth)   # 打印图像文件名
    print(f"save to {args.save_path}")   # 输出保存路径


# 程序入口
if __name__ == '__main__':
    # 配置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="path to save your attention visualized photo. E.g., /home/me/out.jpg")
    parser.add_argument("--data_path", type=str, help="path to your dataset. E.g., dataset/market1501/query")
    parser.add_argument("--pretrain_path", type=str, help="path to your pretrained vit from imagenet or else. E.g., /home/me/cpt/")
    parser.add_argument("--vit_path", type=str, help="path to your trained vanilla vit. E.g., cpt/vit.pth")
    parser.add_argument("--pat_path", type=str, help="path to your trained PAT. E.g., cpt/pat.pth")
    args = parser.parse_args()  # 解析命令行参数
    main(args)   # 调用主函数