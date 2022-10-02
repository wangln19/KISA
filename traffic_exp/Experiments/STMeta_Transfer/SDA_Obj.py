from ast import arg
import os
import yaml
import argparse
import GPUtil
import numpy as np

from UCTB.dataset import TransferDataLoader
from UCTB.model import STMeta_SDA, STMeta
from UCTB.evaluation import metric
from UCTB.train import EarlyStoppingTTest
from UCTB.preprocess.GraphGenerator import GraphGenerator

#####################################################################
# argument parser
parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument('-m', '--model', default='STMeta_v4.model.yml')
parser.add_argument('-sd', '--source_data', default='bike_dc.data.yml')
parser.add_argument('-td', '--target_data', default='bike_dc.data.yml')
parser.add_argument('-tdl', '--target_data_length', default='365', type=str)
parser.add_argument('-pt', '--pretrain', default='True')
parser.add_argument('-ft', '--finetune', default='True')
parser.add_argument('-tr', '--transfer', default='True')
parser.add_argument('-dm', '--dynamic_mode', default='true',type=str)
parser.add_argument('-g', '--gamma', default=0.1,type=float)

args = vars(parser.parse_args())

with open(args['model'], 'r') as f:
    model_params = yaml.load(f)

with open(args['source_data'], 'r') as f:
    sd_params = yaml.load(f)

with open(args['target_data'], 'r') as f:
    td_params = yaml.load(f)

assert sd_params['closeness_len'] == td_params['closeness_len']
assert sd_params['period_len'] == td_params['period_len']
assert sd_params['trend_len'] == td_params['trend_len']

if args["dynamic_mode"].lower() == 'false':
    dynamic_mode = False
else:
    dynamic_mode = True


#####################################################################
# Generate code_version
#group = '{}_to_{}'.format(sd_params["city"],td_params["city"])

code_version = 'SD_{}_TD_{}'.format(args['source_data'].split('.')[0].split('_')[-1],
                                                 args['target_data'].split('.')[0].split('_')[-1])

sub_code_version = 'C{}P{}T{}_TP'.format(sd_params['closeness_len'], sd_params['period_len'], sd_params['trend_len'],
                                             ''.join([e[0].upper() for e in sd_params['graph'].split('-')]))

model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dir')


#####################################################################
# Config data loader

data_loader = TransferDataLoader(sd_params, td_params, model_params, td_data_length=args['target_data_length'])

# loading match traffic
def get_match_traffic(sd_loader,td_loader):
    from sklearn.metrics.pairwise import cosine_similarity

    source_days = len(sd_loader.train_data)//sd_loader.daily_slots
    target_days = len(td_loader.train_data)//td_loader.daily_slots
    source_traffic = sd_loader.train_data[:int(source_days*sd_loader.daily_slots)]
    target_traffic = td_loader.train_data[:int(target_days*td_loader.daily_slots)]

    #print("target_traffic:",target_traffic.shape)

    source_daily_pattern = np.transpose(source_traffic,[1,0]).reshape([sd_loader.station_number,int(sd_loader.daily_slots),-1]).sum(axis=-1)
    target_daily_pattern = np.transpose(target_traffic,[1,0]).reshape([td_loader.station_number,int(td_loader.daily_slots),-1]).sum(axis=-1)
   
    print("source_daily_pattern:",source_daily_pattern.shape)
    match_matrix = cosine_similarity(source_daily_pattern,target_daily_pattern)
    return match_matrix

match_matrix = get_match_traffic(data_loader.sd_loader,data_loader.td_loader)
print("source num node:",data_loader.sd_loader.station_number)
print("target num node:",data_loader.td_loader.station_number)
print("match_matrix:",match_matrix.shape)

pretrain_model_name = 'Pretrain_' + sub_code_version
finetune_model_name = 'Finetune_' + sub_code_version + '_' + str(data_loader.td_loader.train_sequence_len)
transfer_model_name = 'Match_' + sub_code_version + '_' + str(data_loader.td_loader.train_sequence_len)

# build graphs
sd_graph_obj = GraphGenerator(graph=sd_params['graph'],
                           data_loader=data_loader.sd_loader,
                           threshold_distance=sd_params['threshold_distance'],
                           threshold_correlation=sd_params['threshold_correlation'],
                           threshold_interaction=sd_params['threshold_interaction'])

td_graph_obj = GraphGenerator(graph=td_params['graph'],
                           data_loader=data_loader.td_loader,
                           threshold_distance=td_params['threshold_distance'],
                           threshold_correlation=td_params['threshold_correlation'],
                           threshold_interaction=td_params['threshold_interaction'])

deviceIDs = GPUtil.getAvailable(order='memory', limit=2, maxLoad=1, maxMemory=1,
                                includeNan=False, excludeID=[], excludeUUID=[])

if len(deviceIDs) == 0:
    current_device = '-1'
else:
    current_device = str(deviceIDs[0])

sd_model = STMeta(num_node=data_loader.sd_loader.station_number,
                  num_graph=sd_graph_obj.LM.shape[0],
                  external_dim=data_loader.sd_loader.external_dim,
                  tpe_dim=data_loader.sd_loader.tpe_dim,
                  code_version=code_version,
                  model_dir=model_dir_path,
                  save_model_name=pretrain_model_name,
                  gpu_device=current_device,
                  transfer_ratio=0,
                  **sd_params, **model_params)
sd_model.build()

td_model = STMeta_SDA(num_node=data_loader.td_loader.station_number,
                  source_num_node=data_loader.sd_loader.station_number,
                  num_graph=td_graph_obj.LM.shape[0],
                  external_dim=data_loader.td_loader.external_dim,
                  tpe_dim=data_loader.td_loader.tpe_dim,
                  code_version=code_version,
                  model_dir=model_dir_path,
                  save_model_name=transfer_model_name,
                  transfer_ratio=0,
                  gamma=float(args["gamma"]),
                  gpu_device=current_device,
                  **td_params, **model_params)
td_model.build()

sd_de_normalizer = (lambda x: x) if sd_params['normalize'] is False \
                                else data_loader.sd_loader.normalizer.min_max_denormal
td_de_normalizer = (lambda x: x) if td_params['normalize'] is False \
                                else data_loader.td_loader.normalizer.min_max_denormal

print('#################################################################')
print('Source Domain information')
print(sd_params['dataset'], sd_params['city'])
print("source closeness shape:",data_loader.sd_loader.train_closeness.shape)
print("source test_y shape:",data_loader.sd_loader.test_y.shape)
print('Number of trainable variables', sd_model.trainable_vars)
print('Number of training samples', data_loader.sd_loader.train_sequence_len)

print('#################################################################')
print('Target Domain information')
print(td_params['dataset'], td_params['city'])
print("target closeness shape:",data_loader.td_loader.train_closeness.shape)
print("target test_y shape:",data_loader.td_loader.test_y.shape)
print('Number of trainable variables', td_model.trainable_vars)
print('Number of training samples', data_loader.td_loader.train_sequence_len)


sd_transfer_data = data_loader.sd_loader.train_data[-data_loader.td_loader.train_data.shape[0]:, :]

transfer_closeness, \
transfer_period, \
transfer_trend, \
_ = data_loader.sd_loader.st_move_sample.move_sample(sd_transfer_data)

def dynamic_fm(model_name):
    sd_model.load(model_name)
    fm = sd_model.predict(closeness_feature=transfer_closeness,
                            period_feature=transfer_period,
                            trend_feature=transfer_trend,
                            laplace_matrix=sd_graph_obj.LM,
                            external_feature=data_loader.sd_loader.train_ef,
                            output_names=['feature_map'],
                            sequence_length=len(transfer_closeness),
                            cache_volume=sd_params['batch_size'])
    return fm['feature_map']

feature_maps = dynamic_fm(pretrain_model_name)

early_stop = EarlyStoppingTTest(td_params['early_stop_length'], td_params['early_stop_patience'])
best_value = None

# loading model in source domain
sd_model.load(pretrain_model_name)
td_model.load(pretrain_model_name)

for epoch in range(1):
    output = td_model.fit(closeness_feature=data_loader.td_loader.train_closeness,
                            period_feature=data_loader.td_loader.train_period,
                            trend_feature=data_loader.td_loader.train_trend,
                            laplace_matrix=td_graph_obj.LM,
                            target=data_loader.td_loader.train_y,
                            external_feature=data_loader.td_loader.train_ef,
                            source_feature_map=feature_maps,
                            match_matrix=match_matrix,
                            sequence_length=data_loader.td_loader.train_sequence_len,
                            output_names=('loss',),
                            op_names=('train_op',),
                            batch_size=td_params['batch_size'],
                            max_epoch=td_params['max_epoch'],
                            validate_ratio=0.1,
                            early_stop_method='t-test',
                            early_stop_length=td_params['early_stop_length'],
                            early_stop_patience=td_params['early_stop_patience'],
                            save_model_name=transfer_model_name
                            )

    # if dynamic_mode:
    #     feature_maps = dynamic_fm()

    prediction = td_model.predict(closeness_feature=data_loader.td_loader.test_closeness,
                                    period_feature=data_loader.td_loader.test_period,
                                    trend_feature=data_loader.td_loader.test_trend,
                                    laplace_matrix=td_graph_obj.LM,
                                    target=data_loader.td_loader.test_y,
                                    external_feature=data_loader.td_loader.test_ef,
                                    output_names=('prediction',),
                                    sequence_length=data_loader.td_loader.test_sequence_len,
                                    cache_volume=td_params['batch_size'], )

    transfer_prediction = prediction['prediction']
    test_rmse = metric.rmse(prediction=td_de_normalizer(prediction['prediction']),
                            target=td_de_normalizer(data_loader.td_loader.test_y), threshold=0)

   
   
    # validate_error = output[-1]['val_loss'] if validate_mode == 'val' else test_rmse

    # if early_stop.stop(validate_error):
    #     break

    # if best_value is None or best_value > validate_error:
    #     best_value = validate_error
    #     td_model.save(transfer_model_name, global_step=0)

    print(epoch, 'test rmse', test_rmse)



# td_model.load(transfer_model_name)

# prediction = td_model.predict(closeness_feature=data_loader.td_loader.test_closeness,
#                                 period_feature=data_loader.td_loader.test_period,
#                                 trend_feature=data_loader.td_loader.test_trend,
#                                 laplace_matrix=td_graph_obj.LM,
#                                 target=data_loader.td_loader.test_y,
#                                 external_feature=data_loader.td_loader.test_ef,
#                                 output_names=('prediction',),
#                                 sequence_length=data_loader.td_loader.test_sequence_len,
#                                 cache_volume=td_params['batch_size'], )

# transfer_prediction = prediction['prediction']

# test_rmse = metric.rmse(prediction=td_de_normalizer(transfer_prediction),
#                             target=td_de_normalizer(data_loader.td_loader.test_y), threshold=0)

# print('#################################################################')
# print('Target Domain Transfer')
# print("Test RMSE:",test_rmse)