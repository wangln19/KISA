# KISA(Knowledge-inspired Subdomain Adaptation for Cross-Domain Knowledge Transfer)
# Fraud Detection Part
Datasets are unavailable for business consideration
To run the experiment:
1.python share_lstm.py --mode train --filepath XXX --direct True --mark XXX
2.python share_lstm.py --mode generate --filepath XXX --mark XXX
3.python KISA_type.py --mode train --filepath XXX --mark XXX --gamma X
4.python KISA_hour.py --mode train --filepath XXX --mark XXX --gamma X
5.python KISA_type.py --mode generate --filepath XXX --mark XXX --gamma X
6.python KISA_hour.py --mode genarate --filepath XXX --mark XXX --gamma X
7.python KISA.py --mark XXX

Other experiments:
1-2 the same
3-6: change the 'KISA_type.py' to 'KISA_MMD_type.py' / 'KISA_PM_type.py' / 'da_MMD.py' / 'da_coral.py' ...
