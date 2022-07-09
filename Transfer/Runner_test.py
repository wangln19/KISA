import os

# pretrian LZD
# LZD_baseroot = "E://chenliyue//ant//sliding_data//LZD//csv"

# os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets LZD --maxepoch 30".format(LZD_baseroot))

# os.system("python LSTM_pretrain.py --mode generate --baseroot {} --filepath train_2020-01.csv --datasets LZD --maxepoch 30".format(LZD_baseroot))


###########################################
######     pretrian HK LSTM          ###### 
###########################################
HK_baseroot = "E://chenliyue//ant//sliding_data//HK_sample//csv"


###########################################
#######    alpha regular      ######
###########################################
# os.system("python subspace_transfer_regular_alpha.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 120 --gamma 0.01 ".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 120 --gamma 0.01 --mark alpha_regular_test".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 120 --gamma 0.01 --mark wo_regular_test".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 120 --gamma 0.01 --mark regular_test_2".format(HK_baseroot))

#### relu 0.8
# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.8 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark regular_relu_1".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.8 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark regular_relu_2".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.8 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark regular_relu_3".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.8 -mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark regular_relu_4".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.8 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark regular_relu_5".format(HK_baseroot))

#### relu 0.5
# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.5 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot5_1".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.5 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot5_2".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.5 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot5_3".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.5 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot5_4".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.5 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot5_5".format(HK_baseroot))


# #### relu 0.7
# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.7 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot7_1".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.7 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot7_2".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.7 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot7_3".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.7 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot7_4".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.7 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot7_5".format(HK_baseroot))


# #### relu 0.9
# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.9 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot9_1".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.9 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot9_2".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.9 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot9_3".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.9 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot9_4".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_relu.py --max_weight 0.9 --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark relu_dot9_5".format(HK_baseroot))


# os.system("python subspace_transfer_regular_alpha_kl.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark regular_kl".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_kl.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark regular_kl_2".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_kl.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark regular_kl_3".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_kl.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark regular_kl_4".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_kl.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark regular_kl_5".format(HK_baseroot))

# os.system("python subspace_transfer_regular_alpha_freeze.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 120 --gamma 0.01 --mark regular_freeze".format(HK_baseroot))


# ###########################################
# #######     month overlap       ######
# ###########################################

# os.system("python subspace_transfer_overlap_month.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark overlap_month".format(HK_baseroot))

# os.system("python subspace_transfer_overlap_month.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark overlap_month_2".format(HK_baseroot))

# os.system("python subspace_transfer_overlap_month.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark overlap_month_3".format(HK_baseroot))

# os.system("python subspace_transfer_overlap_month.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark overlap_month_4".format(HK_baseroot))

# os.system("python subspace_transfer_overlap_month.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark overlap_month_5".format(HK_baseroot))


# ###########################################
# #######     amount transfer       ######
# ###########################################
# # amount_split_bins 10
# os.system("python subspace_transfer_by_amount.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark amount".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark amount_2".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark amount_3".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark amount_4".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark amount_5".format(HK_baseroot))

# # amount_split_bins 20 
# os.system("python subspace_transfer_by_amount.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_bins 20 --mark amount_bin20".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_bins 20 --mark amount_bin20_2".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_bins 20 --mark amount_bin20_3".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_bins 20 --mark amount_bin20_4".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_bins 20 --mark amount_bin20_5".format(HK_baseroot))

# # amount_split_list [500,5000,50000] 
# os.system("python subspace_transfer_by_amount_list.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_list [500,5000,50000] --mark amount_list_1".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount_list.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_list [500,5000,50000] --mark amount_list_2".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount_list.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_list [500,5000,50000] --mark amount_list_3".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount_list.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_list [500,5000,50000] --mark amount_list_4".format(HK_baseroot))

# os.system("python subspace_transfer_by_amount_list.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_list [500,5000,50000] --mark amount_list_5".format(HK_baseroot))



# amount_split_list [500,5000,20000,50000] 
os.system("python subspace_transfer_by_amount_list.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_list [500,5000,20000,50000] --mark amount_5525_1".format(HK_baseroot))

os.system("python subspace_transfer_by_amount_list.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_list [500,5000,20000,50000] --mark amount_5525_2".format(HK_baseroot))

os.system("python subspace_transfer_by_amount_list.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_list [500,5000,20000,50000] --mark amount_5525_3".format(HK_baseroot))

os.system("python subspace_transfer_by_amount_list.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_list [500,5000,20000,50000] --mark amount_5525_4".format(HK_baseroot))

os.system("python subspace_transfer_by_amount_list.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --amount_split_list [500,5000,20000,50000] --mark amount_5525_5".format(HK_baseroot))



###########################################
#######     save five model       ######
###########################################

# os.system("python subspace_transfer_ensemble.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 ".format(HK_baseroot))

# os.system("python subspace_transfer_ensemble.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01  --mark ensemble2 ".format(HK_baseroot))

# os.system("python subspace_transfer_ensemble.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01  --mark ensemble3 ".format(HK_baseroot))

# os.system("python subspace_transfer_ensemble.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01  --mark ensemble4 ".format(HK_baseroot))


###########################################
#######    parameters search      ######
###########################################
# HK_baseroot = "E://chenliyue//ant//sliding_data//HK_sample//csv"

# os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 50 --lr 0.0001 --mark LR1e4 ".format(HK_baseroot))

