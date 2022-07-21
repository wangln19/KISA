import os


###############################################
# BenchMark DiDi
###############################################
############# Xian #############

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml'
          ' -p train_data_length:2,graph:Distance,closeness_len:24,period_len:0,trend_len:0,mark:LSTM1,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml'
          ' -p train_data_length:4,graph:Distance,closeness_len:24,period_len:0,trend_len:0,mark:LSTM3,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml'
          ' -p train_data_length:8,graph:Distance,closeness_len:24,period_len:0,trend_len:0,mark:LSTM7,MergeIndex:12')


os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p train_data_length:2,closeness_len:24,period_len:0,trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12,mark:V1_1')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p train_data_length:4,closeness_len:24,period_len:0,trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12,mark:V1_3')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_xian.data.yml '
          '-p train_data_length:8,closeness_len:24,period_len:0,trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12,mark:V1_7')

#################################################################
#############    source domian Chengdu       #############

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml'
          ' -p graph:Distance,closeness_len:24,period_len:0,trend_len:0,mark:LSTM,MergeIndex:12')

os.system('python STMeta_Obj.py -m STMeta_v1.model.yml -d didi_chengdu.data.yml '
          '-p closeness_len:24,period_len:0,trend_len:0,graph:Distance-Correlation-Interaction,MergeIndex:12,mark:V1')

