import os

# pretrian LZD
# LZD_baseroot = "E://chenliyue//ant//sliding_data//LZD//csv"

# os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets LZD --maxepoch 30".format(LZD_baseroot))

# os.system("python LSTM_pretrain.py --mode generate --baseroot {} --filepath train_2020-01.csv --datasets LZD --maxepoch 30".format(LZD_baseroot))


###########################################
######     pretrian HK LSTM          ###### 
###########################################
HK_baseroot = "E://chenliyue//ant//sliding_data//HK_sample//csv"
# os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 ".format(HK_baseroot))

os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark lstm1".format(HK_baseroot))

os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark lstm2".format(HK_baseroot))

os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark lstm3".format(HK_baseroot))

os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark lstm4".format(HK_baseroot))

os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark lstm5".format(HK_baseroot))

# os.system("python LSTM_pretrain.py --mode generate --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 200 ".format(HK_baseroot))

###########################################
######      domain adaption          ###### 
###########################################

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 ".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark da1".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark da2".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark da3".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark da4".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark da5".format(HK_baseroot))


###########################################
###### subspace transfer by month    ######
###########################################
# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 ".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark subspace_e2".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark subspace_test".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.001 --mark subspace_e3".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode validate --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.001 --mark subspace_e3".format(HK_baseroot))

# # # # stability  test
# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark month_N1".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark month_N2".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark month_N3".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark month_N4".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark month_N5".format(HK_baseroot))



###########################################
### subspace transfer by month & label ###
###########################################
# os.system("python subspace_transfer_by_label.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 ".format(HK_baseroot))


###########################################
###    subspace transfer by cluster     ###
###########################################
# os.system("python subspace_transfer_by_cluster.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 ".format(HK_baseroot))



###########################################
###    subspace transfer by cluster     ###
###########################################
# os.system("python subspace_transfer_by_cluster_wolabel.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 101 --gamma 0.01 ".format(HK_baseroot))


###########################################
### subspace transfer by cluster mean weight
###########################################
# os.system("python subspace_transfer_mean_weight.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 ".format(HK_baseroot))



###########################################
#######    alpha regular      ######
###########################################
# os.system("python subspace_transfer_regular_alpha.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 ".format(HK_baseroot))



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

