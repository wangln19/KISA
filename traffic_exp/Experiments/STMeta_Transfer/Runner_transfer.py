import os

# ######### pretrain fine-tine
# os.system('python Pretrain_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
#           ' -tdl 2')

# os.system('python Pretrain_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
#           ' -tdl 4')

# os.system('python Pretrain_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
#           ' -tdl 8')

# ######### domain adaptation
# os.system('python Submatch_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
#           ' -tdl 2 --gamma 0.1')

# os.system('python Submatch_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
#           ' -tdl 4 --gamma 0.1')

# os.system('python Submatch_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
#           ' -tdl 8 --gamma 0.1')

######### subdomain adaptation
os.system('python SDA_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
          ' -tdl 2 --gamma 0.1')

# os.system('python SDA_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
#           ' -tdl 4 --gamma 0.1')

# os.system('python SDA_Obj.py -sd didi_chengdu.data.yml -td didi_xian.data.yml'
#           ' -tdl 8 --gamma 0.1')