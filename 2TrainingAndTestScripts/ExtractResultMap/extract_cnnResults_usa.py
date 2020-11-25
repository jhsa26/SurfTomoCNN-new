import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from mpl_toolkits.basemap import Basemap
filepath_vs_cnn     =  './Input/vs_cnn_usa/'
filepath_vs_sws     =  './Input/vs_sws_China/'
os.system('test -d Figs_vs || mkdir Figs_vs')
os.system('rm -rf  layers_vs layers_vs_usa && mkdir layers_vs')
periods = np.array([8,10,12,14,16,18,20,22,24,26,28,30,32,35,40,45,50])
depth = np.array([0,3,10,15,20,30,40,60,80,100,120,150])
count = 0
lay1_sws = [];lay2_sws = [];lay3_sws = [];lay4_sws = [];lay5_sws = [];lay6_sws = [];lay7_sws = [];
lay8_sws = [];lay9_sws = [];lay10_sws = [];lay11_sws = [];lay12_sws = [];lay13_sws = [];

lay1_cnn = [];lay2_cnn = [];lay3_cnn = [];lay4_cnn = [];lay5_cnn = [];lay6_cnn = [];lay7_cnn = [];
lay8_cnn = [];lay9_cnn = [];lay10_cnn = [];lay11_cnn = [];lay12_cnn = [];lay13_cnn = [];

filenames=os.listdir(filepath_vs_sws)
with open('./Input/select.point','r') as f:
    chinaNames=f.read().splitlines()
for key_name in chinaNames:
    temp=key_name.split()
#     temp=key_name.split('.txt')
#     lat,lon=temp[0].split("_")
    lon,lat=key_name.split()
    key_name = lat+"_"+lon+'.txt'
    file_vs_sws = filepath_vs_sws + key_name
    file_vs_cnn = filepath_vs_cnn + key_name
    
    
    if  os.path.exists(file_vs_sws) and os.path.exists(file_vs_cnn):
        count =count +1
        temp_sws = np.loadtxt(file_vs_sws);
        temp_cnn = np.loadtxt(file_vs_cnn);
        if len(temp_sws)>=1 and len(temp_cnn)>=1:
            depth_sws = temp_sws[:,0];vs_sws = temp_sws[:,1]
            depth_cnn=temp_cnn[:,0];vs_cnn = temp_cnn[:,1]
            fl_sws = interp1d(depth_sws, vs_sws, kind='slinear')
            fl_cnn = interp1d(depth_cnn, vs_cnn, kind='slinear')           
            vs_cnn = fl_cnn(depth); vs_sws = fl_sws(depth)
#             print(count,key_name)
            lon = float(lon);lat=float(lat)
    
    
            lay1_sws.append([lon,lat,vs_sws[0]])
            lay2_sws.append([lon,lat,vs_sws[1]])
            lay3_sws.append([lon,lat,vs_sws[2]])
            lay4_sws.append([lon,lat,vs_sws[3]])
            lay5_sws.append([lon,lat,vs_sws[4]])
            lay6_sws.append([lon,lat,vs_sws[5]])
            lay7_sws.append([lon,lat,vs_sws[6]])
            lay8_sws.append([lon,lat,vs_sws[7]])
            lay9_sws.append([lon,lat,vs_sws[8]])
            lay10_sws.append([lon,lat,vs_sws[9]])
            lay11_sws.append([lon,lat,vs_sws[10]])
            lay12_sws.append([lon,lat,vs_sws[11]])
            
            
            lay1_cnn.append([lon,lat,vs_cnn[0]])
            lay2_cnn.append([lon,lat,vs_cnn[1]])
            lay3_cnn.append([lon,lat,vs_cnn[2]])
            lay4_cnn.append([lon,lat,vs_cnn[3]])
            lay5_cnn.append([lon,lat,vs_cnn[4]])
            lay6_cnn.append([lon,lat,vs_cnn[5]])
            lay7_cnn.append([lon,lat,vs_cnn[6]])
            lay8_cnn.append([lon,lat,vs_cnn[7]])
            lay9_cnn.append([lon,lat,vs_cnn[8]])
            lay10_cnn.append([lon,lat,vs_cnn[9]])
            lay11_cnn.append([lon,lat,vs_cnn[10]])
            lay12_cnn.append([lon,lat,vs_cnn[11]])
            

print(filepath_vs_sws)
print(count)


lay1=np.array(lay1_sws);np.savetxt('./layers_vs/lay1_sws.txt',lay1,fmt="%10.5f")
lay2=np.array(lay2_sws);np.savetxt('./layers_vs/lay2_sws.txt',lay2,fmt="%10.5f")
lay3=np.array(lay3_sws);np.savetxt('./layers_vs/lay3_sws.txt',lay3,fmt="%10.5f")
lay4=np.array(lay4_sws);np.savetxt('./layers_vs/lay4_sws.txt',lay4,fmt="%10.5f")
lay5=np.array(lay5_sws);np.savetxt('./layers_vs/lay5_sws.txt',lay5,fmt="%10.5f")
lay6=np.array(lay6_sws);np.savetxt('./layers_vs/lay6_sws.txt',lay6,fmt="%10.5f")
lay7=np.array(lay7_sws);np.savetxt('./layers_vs/lay7_sws.txt',lay7,fmt="%10.5f")
lay8=np.array(lay8_sws);np.savetxt('./layers_vs/lay8_sws.txt',lay8,fmt="%10.5f")
lay9=np.array(lay9_sws);np.savetxt('./layers_vs/lay9_sws.txt',lay9,fmt="%10.5f")
lay10=np.array(lay10_sws);np.savetxt('./layers_vs/lay10_sws.txt',lay10,fmt="%10.5f")
lay11=np.array(lay11_sws);np.savetxt('./layers_vs/lay11_sws.txt',lay11,fmt="%10.5f")
lay12=np.array(lay12_sws);np.savetxt('./layers_vs/lay12_sws.txt',lay12,fmt="%10.5f")

lay1=np.array(lay1_cnn);np.savetxt('./layers_vs/lay1_cnn.txt',lay1,fmt="%10.5f")
lay2=np.array(lay2_cnn);np.savetxt('./layers_vs/lay2_cnn.txt',lay2,fmt="%10.5f")
lay3=np.array(lay3_cnn);np.savetxt('./layers_vs/lay3_cnn.txt',lay3,fmt="%10.5f")
lay4=np.array(lay4_cnn);np.savetxt('./layers_vs/lay4_cnn.txt',lay4,fmt="%10.5f")
lay5=np.array(lay5_cnn);np.savetxt('./layers_vs/lay5_cnn.txt',lay5,fmt="%10.5f")
lay6=np.array(lay6_cnn);np.savetxt('./layers_vs/lay6_cnn.txt',lay6,fmt="%10.5f")
lay7=np.array(lay7_cnn);np.savetxt('./layers_vs/lay7_cnn.txt',lay7,fmt="%10.5f")
lay8=np.array(lay8_cnn);np.savetxt('./layers_vs/lay8_cnn.txt',lay8,fmt="%10.5f")
lay9=np.array(lay9_cnn);np.savetxt('./layers_vs/lay9_cnn.txt',lay9,fmt="%10.5f")
lay10=np.array(lay10_cnn);np.savetxt('./layers_vs/lay10_cnn.txt',lay10,fmt="%10.5f")
lay11=np.array(lay11_cnn);np.savetxt('./layers_vs/lay11_cnn.txt',lay11,fmt="%10.5f")
lay12=np.array(lay12_cnn);np.savetxt('./layers_vs/lay12_cnn.txt',lay12,fmt="%10.5f")
# before run the cell, please run the matlab script "MainSmoothResults.m" to smooth the results and plot.
os.system("matlab -nojvm -nodisplay -nosplash -nodesktop < MainSmoothResults.m")
print("ending matlab")
os.system("mv layers_vs layers_vs_usa")