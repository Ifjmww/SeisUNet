# 按照伍老师的分块方式和拼接方式，进行真实数据预测

import os
from utils.tools import choose_model, save_pred_picture, load_pred_data
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# set gaussian weights in the overlap bounaries
def getMask(overlap, n1, n2, n3):
    sc = np.zeros((n1, n2, n3), dtype=np.single)
    sc = sc + 1
    sp = np.zeros((overlap), dtype=np.single)
    sig = overlap / 4
    sig = 0.5 / (sig * sig)
    for ks in range(overlap):
        ds = ks - overlap + 1
        sp[ks] = np.exp(-ds * ds * sig)
    for k1 in range(overlap):
        for k2 in range(n2):
            for k3 in range(n3):
                sc[k1][k2][k3] = sp[k1]
                sc[n1 - k1 - 1][k2][k3] = sp[k1]
    for k1 in range(n1):
        for k2 in range(overlap):
            for k3 in range(n3):
                sc[k1][k2][k3] = sp[k2]
                sc[k1][n3 - k2 - 1][k3] = sp[k2]
    for k1 in range(n1):
        for k2 in range(n2):
            for k3 in range(overlap):
                sc[k1][k2][k3] = sp[k3]
                sc[k1][k2][n3 - k3 - 1] = sp[k3]
    # return np.transpose(sc)
    return sc


def pred_f3wu_Gaussian(args):
    print("============================== pred_f3wu_Gaussian ==============================")
    # load and create model
    model = choose_model(args)
    model = model.to(args.device)

    model_path = './EXP/' + args.model_type + '/' + args.exp + '/models/' + args.pretrained_model_name

    model.load_state_dict(torch.load(model_path))
    print("Loaded model from disk")

    # training image dimensions
    n1, n2, n3 = 128, 128, 128

    # 加载数据
    args.pred_data_name = 'f3wu'
    gx = load_pred_data(args)

    m1, m2, m3 = gx.shape[0], gx.shape[1], gx.shape[2]

    args.overlap = 12  # overlap width
    c1 = np.round((m1 + args.overlap) / (n1 - args.overlap) + 0.5)
    c2 = np.round((m2 + args.overlap) / (n2 - args.overlap) + 0.5)
    c3 = np.round((m3 + args.overlap) / (n3 - args.overlap) + 0.5)

    c1 = int(c1)
    c2 = int(c2)
    c3 = int(c3)

    p1 = (n1 - args.overlap) * c1 + args.overlap
    p2 = (n2 - args.overlap) * c2 + args.overlap
    p3 = (n3 - args.overlap) * c3 + args.overlap

    gp = np.zeros((p1, p2, p3), dtype=np.single)
    gy = np.zeros((p1, p2, p3), dtype=np.single)
    mk = np.zeros((p1, p2, p3), dtype=np.single)
    gs = np.zeros((1, 1, n1, n2, n3), dtype=np.single)
    gp[0:m1, 0:m2, 0:m3] = gx
    sc = getMask(args.overlap, n1, n2, n3)

    print('>>>Start Predicting<<<')

    total_iterations = c1 * c2 * c3
    progress_bar = tqdm(total=total_iterations, desc='[Pred]', unit='it')

    for k1 in range(c1):
        for k2 in range(c2):
            for k3 in range(c3):

                b1 = k1 * n1 - k1 * args.overlap
                e1 = b1 + n1
                b2 = k2 * n2 - k2 * args.overlap
                e2 = b2 + n2
                b3 = k3 * n3 - k3 * args.overlap
                e3 = b3 + n3
                # gp[b1:e1, b2:e2, b3:e3]
                temp = gp[b1:e1, b2:e2, b3:e3]
                # 正则化
                temp_trans = np.transpose(temp)
                temp_m = np.mean(temp_trans)
                temp_s = np.std(temp_trans)
                temp_trans = (temp_trans - temp_m) / temp_s
                gs[0, 0, :, :, :] = temp_trans[:, :, :]

                if args.in_channels == 1:
                    inputs = torch.from_numpy(gs).to(args.device)
                elif args.in_channels == 2:

                    gs_G_cl = np.gradient(gs, axis=3)
                    gs_G_il = np.gradient(gs, axis=4)
                    gs_G = np.concatenate((gs_G_cl, gs_G_il), axis=1)
                    inputs = torch.from_numpy(gs_G).to(args.device)
                else:
                    raise ValueError("in_channels should be 1 or 2 !")
                y = model(inputs)
                outputs = y.argmax(axis=1)
                outputs = outputs.detach().cpu().numpy()
                outputs = np.transpose(np.squeeze(outputs))

                gy[b1:e1, b2:e2, b3:e3] = gy[b1:e1, b2:e2, b3:e3] + outputs[:, :, :] * sc
                mk[b1:e1, b2:e2, b3:e3] = mk[b1:e1, b2:e2, b3:e3] + sc
                progress_bar.update(1)
    progress_bar.close()
    gy = gy / mk
    gy = gy[0:m1, 0:m2, 0:m3]

    print("---Start Save results  ······")
    save_path = './EXP/' + args.model_type + '/' + args.exp + '/results/pred/' + args.pred_data_name + '_f3wuGuassian/'
    if not os.path.exists(save_path + '/numpy/'):
        os.makedirs(save_path + '/numpy/')
    if not os.path.exists(save_path + '/picture/'):
        os.makedirs(save_path + '/picture/')
    np.save(save_path + '/numpy/' + args.pred_data_name + '.npy', np.transpose(gy))

    save_pred_picture(np.transpose(gx), np.transpose(gy), save_path + '/picture/', args.pred_data_name)
