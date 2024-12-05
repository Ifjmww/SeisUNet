# SeisUNet:3D Seismic Fault Segmentation Using a Customized U-Net
import argparse
from utils.train import train, valid
from utils.test_Gaussian import pred_Gaussian
from utils.tools import save_args_info
import warnings

warnings.filterwarnings('ignore')


def add_args():
    parser = argparse.ArgumentParser(description="SeisUNet")

    parser.add_argument("--exp", default="test", type=str, help="Name of each run")
    parser.add_argument("--device", default='cuda:0', type=str, help="GPU id for training")
    parser.add_argument("--mode", default='train', choices=['train', 'valid_only', 'pred'], type=str,
                        help='network run mode')
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--batch_size_not_train", default=1, type=int, help="number of batch size when not training")
    parser.add_argument("--epochs", default=500, type=int, help="max number of training epochs")
    parser.add_argument("--train_path", default="/data/train/", type=str, help="dataset directory")
    parser.add_argument("--valid_path", default="/data/valid/", type=str, help="dataset directory")
    parser.add_argument("--pred_path", default="/data/pred/", type=str, help="dataset directory")
    parser.add_argument("--in_channels", default=2, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
    parser.add_argument("--loss_func", default="dice", choices=['dice', 'cross_with_weight'], type=str,
                        help="choose loss function")
    parser.add_argument("--lr_sc", default="Ex_LR", choices=['Ex_LR', 'Cos_LR'], type=str,
                        help="optimization algorithm")
    parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
    parser.add_argument("--optim_lr", default=0.001, type=float, help="optimization learning rate")
    parser.add_argument("--l2_reg", default=0.1, type=float, help="L2 reg")
    parser.add_argument("--workers", default=0, type=int, help="number of workers")
    parser.add_argument("--model_type", default='LP_All_SO',
                        choices=['LP_All', 'LP_All_SO', 'LP_T_All', 'LP_T_All_SO', 'MP_All_SO', 'LP_P1_SO', 'LP_P12_SO',
                                 'LP_P3_SO'],
                        type=str,
                        help='network run mode')
    parser.add_argument("--pred_data_name", default="kerry",
                        choices=['kerry', 'kerry_mini', 'opunake', 'f3'], type=str,
                        help="pretrained data name")
    parser.add_argument("--pretrained_model_name", default="SeisUNet_BEST.pth", type=str, help="pretrained model name")
    # 高斯滤波器拼接时将要用到的参数
    parser.add_argument('--overlap', default=0.25, type=float, help='pred‘s overlap')
    parser.add_argument('--threshold', default=0.5, type=float, help='Classification threshold')
    parser.add_argument('--sigma', default=0.0, type=float, help='Gaussian filter sigma')

    args = parser.parse_args()

    print()
    print(">>>============= args ====================<<<")
    print()
    print(args)  # print command line args
    print()
    print(">>>=======================================<<<")

    return args


def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'valid_only':
        valid(args)
    elif args.mode == 'pred':
        pred_Gaussian(args)
    else:
        raise ValueError("Only ['train', 'valid_only', 'pred'] mode is supported.")
    save_args_info(args)


if __name__ == "__main__":
    args = add_args()
    main(args)
