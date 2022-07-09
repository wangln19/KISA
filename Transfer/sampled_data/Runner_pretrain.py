import os

# # pretrian HK 80% samples
HK_baseroot = "E://chenliyue//ant//sliding_data//HK_sample_80//csv"

# # os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 80prct".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 80prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 80prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.001 --mark 80prct --para _1e3".format(HK_baseroot))

os.system("python subspace_transfer_by_cluster.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 80prct".format(HK_baseroot))


# # pretrian HK 60% samples
HK_baseroot = "E://chenliyue//ant//sliding_data//HK_sample_60//csv"

# # os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 60prct".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 60prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 60prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.001 --mark 60prct --para _1e3".format(HK_baseroot))

os.system("python subspace_transfer_by_cluster.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 60prct".format(HK_baseroot))


# # pretrian HK 40% samples
HK_baseroot = "E://chenliyue//ant//sliding_data//HK_sample_40//csv"

# # os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 40prct".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 40prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 40prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.001 --mark 40prct --para _1e3".format(HK_baseroot))

os.system("python subspace_transfer_by_cluster.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 40prct".format(HK_baseroot))


# # pretrian HK 20% samples
HK_baseroot = "E://chenliyue//ant//sliding_data//HK_sample_20//csv"

# # os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 20prct".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 20prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 20prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.001 --mark 20prct --para _1e3".format(HK_baseroot))

os.system("python subspace_transfer_by_cluster.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 20prct".format(HK_baseroot))


# # pretrian HK 10% samples
HK_baseroot = "E://chenliyue//ant//sliding_data//HK_sample_10//csv"

# os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 10prct".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 10prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 10prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.001 --mark 10prct --para _1e3".format(HK_baseroot))

os.system("python subspace_transfer_by_cluster.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 10prct".format(HK_baseroot))


# # pretrian HK 5% samples
HK_baseroot = "E://chenliyue//ant//sliding_data//HK_sample_5//csv"

# os.system("python LSTM_pretrain.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 5prct".format(HK_baseroot))

# os.system("python domain_adaption.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --mark 5prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 5prct".format(HK_baseroot))

# os.system("python subspace_transfer.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.001 --mark 5prct --para _1e3".format(HK_baseroot))

os.system("python subspace_transfer_by_cluster.py --mode train --baseroot {} --filepath train_2020-01.csv --datasets HK --maxepoch 100 --gamma 0.01 --mark 5prct".format(HK_baseroot))
