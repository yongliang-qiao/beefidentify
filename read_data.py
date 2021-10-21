import numpy as np
A=np.load('/home/yqiao/PycharmProjects/cattleseq/five-video-classification-methods-master/data/sequences/v_ApplyEyeMakeup_g01_c03-40-features.npy')
#print(A['arr_0'])



use_neutral=True
if not use_neutral:
        print("Reading genders...")