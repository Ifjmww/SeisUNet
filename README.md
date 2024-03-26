# SeisUNet for Seismic Fault Segmentation

requirement
```angular2html
pip install pandas==1.3.5 torchsummary==1.5.1 scikit-learn==0.24.2 openpyxl==3.0.10
```


train-test
```angular2html
python main.py --mode train --model_type LongPool_T_SO --exp test --epochs 2 --train_path ./data_3D_20/train/ --valid_path ./data_3D_20/valid/ --val_every 1 --in_channels 1 --batch_size 2
```

train
```angular2html
python main.py --mode train --model_type LongPool_SO --exp 0630_d800+NF_val24_e50_LP_SO_dice --epochs 50 --train_path ./data_3D_800/train/ --valid_path ./data_3D_800/valid/ --val_every 1 --in_channels 1 --workers 8 --batch_size 2
```

valid_only
```angular2html
python main.py --mode valid_only --model_type LongPool_T_SO --exp 0627_d800+NF_val96_e50_LPT_SO_dice --valid_path ./data_3D_800+NFF3/valid/ --in_channels 1
```

test
```angular2html
python main.py --mode pred --model_type LongPool_T_SO --pred_mode Gauss --exp 0629_d800+NF_val24_e50_LPT_SO_dice --in_channels 1  --pred_data_name f3_2023_demo_cut
```

test_circle
```angular2html
python main.py --mode pred --model_type LongPool_T_SO --exp 0627_d800+NF_train48_val48_e50_LPT_SO_dice --in_channels 1 --pred_data_name f3_2023_demo_cut --pred_mode Circle
```