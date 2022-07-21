import os
import nni
import numpy as np

from UCTB.dataset import NodeTrafficLoader
from UCTB.model import STMeta_V1
from UCTB.evaluation import metric
from UCTB.model_unit import GraphBuilder
from UCTB.preprocess import is_work_day_china

model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dir')


def cpt_stmeta_param_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Argument Parser")
    # data source
    parser.add_argument('--Dataset', default='DiDi')
    parser.add_argument('--City', default='Xian')
    # network parameter
    parser.add_argument('--CT', default='6', type=int)
    parser.add_argument('--PT', default='7', type=int)
    parser.add_argument('--TT', default='4', type=int)
    parser.add_argument('--K', default='1', type=int)
    parser.add_argument('--L', default='1', type=int)
    parser.add_argument('--Graph', default='Distance-Interaction-Correlation')
    parser.add_argument('--GLL', default='1', type=int)
    parser.add_argument('--LSTMUnits', default='64', type=int)
    parser.add_argument('--GALUnits', default='64', type=int)
    parser.add_argument('--GALHeads', default='2', type=int)
    parser.add_argument('--DenseUnits', default='32', type=int)
    parser.add_argument('--Normalize', default='True', type=str)
    # Training data parameters
    parser.add_argument('--DataRange', default='All')
    parser.add_argument('--TrainDays', default='All')
    # Graph parameter
    parser.add_argument('--TC', default='0.7', type=float)
    parser.add_argument('--TD', default='3000', type=float)
    parser.add_argument('--TI', default='100', type=float)
    # training parameters
    parser.add_argument('--Epoch', default='5000', type=int)
    parser.add_argument('--Train', default='True', type=str)
    parser.add_argument('--lr', default='1e-4', type=float)
    parser.add_argument('--ESlength', default='200', type=int)
    parser.add_argument('--patience', default='0.1', type=float)
    parser.add_argument('--BatchSize', default='128', type=int)
    # device parameter
    parser.add_argument('--Device', default='0,1', type=str)
    # version control
    parser.add_argument('--Group', default='Xian')
    parser.add_argument('--CodeVersion', default='ParamTuner')
    return parser


parser = cpt_stmeta_param_parser()
args = vars(parser.parse_args())

args.update(nni.get_next_parameter())

model_dir = os.path.join(model_dir_path, args['Group'])
code_version = 'CPT_STMeta_{}_K{}L{}_{}'.format(''.join([e[0] for e in args['Graph'].split('-')]),
                                                      args['K'], args['L'], args['CodeVersion'] + nni.get_sequence_id())

# Config data loader
data_loader = NodeTrafficLoader(dataset=args['Dataset'], city=args['City'],
                                  data_range=args['DataRange'], train_data_length=args['TrainDays'],
                                  test_ratio=0.1,
                                  C_T=int(args['CT']), P_T=int(args['PT']), T_T=int(args['TT']),
                                  TI=args['TI'], TD=args['TD'], TC=args['TC'],
                                  normalize=True if args['Normalize'] == 'True' else False,
                                  graph=args['Graph'], with_lm=True)

de_normalizer = None if args['Normalize'] == 'False' else data_loader.normalizer.min_max_denormal

CPT_STMeta_Obj = STMeta_V1(num_node=data_loader.station_number,
                                  num_graph=data_loader.LM.shape[0],
                                  external_dim=data_loader.external_dim,
                                  C_T=int(args['CT']), P_T=int(args['PT']), T_T=int(args['TT']),
                                  GCN_K=int(args['K']),
                                  GCN_layers=int(args['L']),
                                  GCLSTM_layers=int(args['GLL']),
                                  gal_units=int(args['GALUnits']),
                                  gal_num_heads=int(args['GALHeads']),
                                  num_hidden_units=int(args['LSTMUnits']),
                                  num_filter_conv1x1=int(args['DenseUnits']),
                                  lr=float(args['lr']),
                                  code_version=code_version,
                                  model_dir=model_dir,
                                  GPU_DEVICE=args['Device'])

CPT_STMeta_Obj.build()

print(args['Dataset'], args['City'], code_version)
print('Number of trainable variables', CPT_STMeta_Obj.trainable_vars)

# Training
if args['Train'] == 'True':
    CPT_STMeta_Obj.fit(closeness_feature=data_loader.train_closeness,
                       period_feature=data_loader.train_period,
                       trend_feature=data_loader.train_trend,
                       laplace_matrix=data_loader.LM,
                       target=data_loader.train_y,
                       external_feature=data_loader.train_ef,
                       early_stop_method='t-test',
                       early_stop_length=int(args['ESlength']),
                       early_stop_patience=float(args['patience']),
                       batch_size=int(args['BatchSize']),
                       max_epoch=int(args['Epoch']))

CPT_STMeta_Obj.load(code_version)

# Evaluate
test_error = CPT_STMeta_Obj.evaluate(closeness_feature=data_loader.test_closeness,
                                     period_feature=data_loader.test_period,
                                     trend_feature=data_loader.test_trend,
                                     laplace_matrix=data_loader.LM,
                                     target=data_loader.test_y,
                                     external_feature=data_loader.test_ef,
                                     cache_volume=int(args['BatchSize']),
                                     metrics=[metric.rmse, metric.mape],
                                     de_normalizer=de_normalizer,
                                     threshold=0)

print('Test result', test_error)

val_loss = CPT_STMeta_Obj.load_event_scalar('val_loss')

best_val_loss = min([e[-1] for e in val_loss])

nni.report_final_result({
    'default': best_val_loss,
    'test-rmse': test_error[0],
    'test-mape': test_error[1]
})
