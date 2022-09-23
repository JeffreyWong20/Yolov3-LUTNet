#================================================================
#
#   File name   : conversion.py
#   Created date: 2020-08-30
#   Description : functions that used to convert weights back to -1,1 with gamma modification
#
#================================================================

import h5py
import numpy as np
import sys

from shutil import copyfile

def conversion():

    #copyfile(output_path + "/2_residuals.h5", output_path + "/bnn.h5") 				 # create pretrained.h5 using datastructure from dummy.h5
    # copyfile(output_path + "/2_residuals.h5", output_path + "/pretrained_pruned.h5") # create pretrained.h5 using datastructure from dummy.h5
    # bl = h5py.File(output_path + "/2_residuals.h5", 'r')
    # pretrained = h5py.File(output_path + "/pretrained_pruned.h5", 'r+')


    copyfile("2_residuals.h5", "baseline_reg.h5") # create pretrained.h5 using datastructure from dummy.h5
    bl = h5py.File("2_residuals.h5", 'r')
    pretrained = h5py.File("baseline_reg.h5", 'r+')


    normalisation="l2"

    # for key in bl['model_weights'].attrs['layer_names']:
    for key in bl:
        if 'binary_conv' in key:	
            print("Layer: ",key)
            bl_w1_ori = bl[key][key]["Variable_1:0"]
            bl_w1 = bl[key][key]["Variable_1:0"][()]
            bl_gamma = bl[key][key]["Variable:0"][()]

            pret_w1 = pretrained[key][key]["Variable_1:0"]
            pret_gamma = pretrained[key][key]["Variable:0"]

            double_bl_w1 = bl_w1 
            half_gamma = bl_gamma * 0.5

            pret_gamma[...] = np.array(half_gamma,dtype=float)
            pret_w1[...] = np.array(double_bl_w1,dtype=float)
            pret_w1_data = pret_w1[()]
            print("ORI_")
            print(bl_w1_ori)
            print("FIX_")
            print(pret_w1)
            print("ORI")
            print(bl_w1)
            print("FIX")
            print(pret_w1_data)
			
            #p_gamma[...] = np.array(bl_gamma)
        if 'residual_sign' in key:	
            try:
                bl_means = bl[key][key]["means:0"][()]
                pret_means = pretrained[key][key]["means:0"]
                half_bl_means = bl_means * 0.5
                pret_means[...] = np.array(half_bl_means, dtype=float)
                print("<---------READIN------->")
                print("Layer: ", key)
                print("Converted")
                print(pret_means)
                print("bl_means")
                print(bl_means)

            except:
                print("Layer: ",key, "NO MEAN")
                print("Layer: ",bl[key]['residual_sign_4'].keys() )
                bl_means = bl[key]['residual_sign_4']["means:0"][()]
                pret_means = pretrained[key]['residual_sign_4']["means:0"]
                half_bl_means = bl_means * 0.5
                pret_means[...] = np.array(half_bl_means, dtype=float)

        

    pretrained.close()

    #copyfile("pretrained_pruned.h5", output_path + "/2_residuals.h5") 

conversion()