# 真实数据预测，分块，拼接时用高斯模糊边界，避免边界感太强

import os
from utils.tools import choose_model, save_pred_picture, load_pred_data
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt


def sliding_window_prediction(input_data, block_size, overlap, model, args):
    # 输入数据的尺寸
    input_shape = input_data.shape
    print('input_shape', input_shape)
    # 切块大小和步长
    block_shape = np.array(block_size)
    step = (1 - overlap) * block_shape

    # 计算需要切割成的块数
    num_blocks = np.ceil(input_shape / step).astype(int)

    # 初始化预测结果和权重矩阵
    sliding_shape = np.array(((num_blocks[0] - 1) * step[0] + block_shape[0],
                              (num_blocks[1] - 1) * step[1] + block_shape[1],
                              (num_blocks[2] - 1) * step[2] + block_shape[2])).astype(int)
    print('sliding_shape', sliding_shape)
    sliding_data = np.zeros(sliding_shape)

    sliding_data[0:input_shape[0], 0:input_shape[1], 0:input_shape[2]] = input_data

    output = np.zeros(sliding_shape)
    weight_map = np.zeros(sliding_shape)

    total_iterations = num_blocks[0] * num_blocks[1] * num_blocks[2]
    progress_bar = tqdm(total=total_iterations, desc='[Pred]', unit='it')

    # 滑动窗口切块和预测
    for i in range(num_blocks[0]):
        for j in range(num_blocks[1]):
            for k in range(num_blocks[2]):
                # 计算当前块的起始和结束位置
                start = (step * np.array([i, j, k])).astype(int)
                end = (start + block_shape).astype(int)
                # 裁剪当前块的数据
                block = sliding_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                block = block.reshape((1, 1, block.shape[0], block.shape[1], block.shape[2]))

                block_mean = np.mean(block)
                block_std = np.std(block)
                block_normal = (block - block_mean) / block_std

                # 进行预测（这里假设使用你的语义分割模型进行预测

                if args.in_channels == 1:
                    # 单通道，就原始数据自己
                    input_block = torch.from_numpy(block_normal).to(args.device)
                elif args.in_channels == 2:
                    # 双通道，仅包含两个方向的导数
                    gs_G_cl = np.gradient(block, axis=3)
                    gs_G_il = np.gradient(block, axis=4)
                    # 梯度数据正则化
                    gs_G_cl_normal = (gs_G_cl - np.mean(gs_G_cl)) / np.std(gs_G_cl)
                    gs_G_il_normal = (gs_G_il - np.mean(gs_G_il)) / np.std(gs_G_il)
                    # 拼接，先crossline，再inline
                    gs_G = np.concatenate((gs_G_cl_normal, gs_G_il_normal), axis=1)
                    input_block = torch.from_numpy(gs_G).to(args.device).float()
                else:  # args.in_channels == 3:
                    # 三通道，两个方向的梯度数据+原始数据
                    gs_G_cl = np.gradient(block, axis=3)
                    gs_G_il = np.gradient(block, axis=4)
                    # 梯度数据正则化
                    gs_G_cl_normal = (gs_G_cl - np.mean(gs_G_cl)) / np.std(gs_G_cl)
                    gs_G_il_normal = (gs_G_il - np.mean(gs_G_il)) / np.std(gs_G_il)
                    gs_G = np.concatenate((gs_G_cl_normal, gs_G_il_normal, block_normal), axis=1)
                    input_block = torch.from_numpy(gs_G).to(args.device).float()
                block_prediction = model(input_block)

                block_prediction = block_prediction[:, 1, :, :, :]
                # block_prediction = block_prediction.argmax(axis=1)
                block_prediction = block_prediction.detach().cpu().numpy()
                block_prediction = np.squeeze(block_prediction)

                # 计算当前块的权重矩阵
                weight_map[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += 1

                # 将当前块的预测结果叠加到输出中
                output[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += block_prediction
                progress_bar.update(1)
    progress_bar.close()

    # 根据权重矩阵对预测结果进行归一化
    output /= weight_map

    # 使用高斯滤波器对边界进行平滑
    smoothed_output = gaussian_filter(output, sigma=args.sigma)

    return smoothed_output[0:input_shape[0], 0:input_shape[1], 0:input_shape[2]]


def pred_Gaussian(args):
    print("============================== pred_Gaussian ==============================")
    input_data = load_pred_data(args)  # 输入数据
    block_size = (128, 128, 128)  # 切块大小
    overlap = args.overlap  # 重叠率

    # 使用训练好的模型进行预测
    model = choose_model(args)
    model = model.to(args.device)
    model_path = './EXP/' + args.model_type + '/' + args.exp + '/models/' + args.pretrained_model_name
    model.load_state_dict(torch.load(model_path))
    print("Loaded model from disk")

    model.eval()
    # 调用滑动窗口预测函数
    output_data = sliding_window_prediction(input_data, block_size, overlap, model, args)

    threshold = args.threshold
    output_data[output_data > threshold] = 1
    output_data[output_data <= threshold] = 0

    print("---Start Save results  ······")
    save_path = './EXP/' + args.model_type + '/' + args.exp + '/results/pred/' + args.pred_data_name + '/'
    if not os.path.exists(save_path + '/numpy/'):
        os.makedirs(save_path + '/numpy/')
    if not os.path.exists(save_path + '/picture/'):
        os.makedirs(save_path + '/picture/')
    np.save(save_path + '/numpy/' + args.pred_data_name + '.npy', output_data)

    save_pred_picture(input_data, output_data, save_path + '/picture/', args.pred_data_name)
    print("Finish!!!")
