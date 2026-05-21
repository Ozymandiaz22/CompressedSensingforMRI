import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import ismrmrd 
import twixtools

##close all files



if len(sys.argv) >= 2:
    directory = sys.argv[1]
else:
    directory = "C:/Users/osman/Documents/FYP Datasets/Batch 1/HV15_MRD/5927"

##find the mrd file in the directory
mrd_files = [f for f in os.listdir(directory) if f.endswith(".MRD")]
if not mrd_files:
    print("No MRD file found in the specified directory.")
mrd_file = mrd_files[0]
mrdpath = os.path.join(directory, mrd_file).replace('\\', '/')
print(f"MRD file found: {mrdpath}")


dat_files = [f for f in os.listdir(directory) if f.endswith(".dat")]
if not dat_files:
    print("No .dat file found in the specified directory.")
print(os.path.join(directory, dat_files[0]).replace('\\', '/'))

twixread = twixtools.read_twix(mrdpath,parse_prot=False)




# ##print the mrd file path
# print(f"MRD file found: {os.path.join(directory, mrd_file).replace('/', '/')}")
# path = os.path.join(directory, mrd_file).replace('/', '/')
# ###read the mrd file
# mrd_data = ismrmrd.Dataset(path)
# ##get the header information
# header = mrd_data.read_xml_header()
# print("Header information:")
# print(header)

