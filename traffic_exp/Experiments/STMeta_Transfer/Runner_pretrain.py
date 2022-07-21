import os

os.system('python Pretrain_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
          ' -tdl 2')

os.system('python Pretrain_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
          ' -tdl 4')

os.system('python Pretrain_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
          ' -tdl 8')