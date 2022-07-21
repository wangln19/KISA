import os

#############################################
# BenchMark Bike
#############################################

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_nyc.data.yml '
          '-p graph:Distance-Correlation-Interaction')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_chicago.data.yml '
          '-p graph:Distance-Correlation-Interaction')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d bike_dc.data.yml '
          '-p graph:Distance-Correlation-Interaction')


# ###############################################
# # BenchMark DiDi
# ###############################################

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_xian.data.yml '
          '-p graph:Distance-Correlation-Interaction')

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d didi_chengdu.data.yml '
          '-p mark:V0')

###############################################
# BenchMark Metro
###############################################

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_chongqing.data.yml '
          '-p graph:Distance')
#
os.system('python STMeta_Obj.py -m STMeta_v0.model.yml -d metro_shanghai.data.yml '
          '-p graph:Distance-Correlation-Line')

###############################################
# BenchMark ChargeStation
###############################################

os.system('python STMeta_Obj.py -m STMeta_v0.model.yml'
          ' -d chargestation_beijing.data.yml -p graph:Distance,mark:V0')
