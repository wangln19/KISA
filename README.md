# KISA(Knowledge-inspired Subdomain Adaptation for Cross-Domain Knowledge Transfer)
# Fraud Detection Part (in the Transfer File)
Datasets are unavailable for business consideration  

To run the KISA:  

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

The detailed impleament can be found in the paper.  

Abstract:  

Most state-of-the-art deep domain adaptation techniques align source and target samples in a global fashion. That is, after alignment, each source sample is expected to become similar to any target sample. However, global alignment may not always be optimal or necessary in practice. For example, consider cross-domain fraud detection, where there are two types of transactions: credit and non-credit. Aligning credit and non-credit transactions separately may yield better performance than global alignment, as credit transactions are unlikely to exhibit patterns similar to non-credit transactions. To enable such fine-grained domain adaption, we propose a novel Knowledge-Inspired Subdomain Adaptation (KISA) framework. In particular, (1) We provide the theoretical insight that KISA minimizes the shared expected loss which is the premise for the success of domain adaptation methods. (2) We propose the knowledge-inspired subdomain division problem that
plays a crucial role in fine-grained domain adaption. (3) We design a knowledge fusion network to exploit diverse domain knowledge. Extensive experiments emonstrate that KISA achieves remarkable results on fraud detection and traffic demand prediction tasks.
