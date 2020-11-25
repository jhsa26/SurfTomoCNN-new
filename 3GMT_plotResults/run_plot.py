import numpy as np
import os
depth=np.array([0,3,10,15,20,30,40,60,80,100,120])
cpt=["2.0/3.6","2.5/3.5","3.1/3.7","3.2/3.8","3.2/3.9","3.2/4.2","3.4/4.6","3.7/4.6","4.15/4.7","4.2/4.8","4.05/4.75","4.1/4.8","4.1/4.7"]
#     0         3         10         15       20        30          40       3.6 60   80         100      120      150       180\n",
os.system("rm -rf Figures")
os.system("mkdir  Figures")
path="./layers_vs_usa/"
path1="./layers_vs_usa/"
path2="./layers_vs_usa_tibet/"
for i in range(len(depth)):
    dep = str(depth[i])
    cptrange = cpt[i]
    datacnn1=path1+"lay"+str(i+1)+"_cnn.txt"
    datacnn2=path2+"lay"+str(i+1)+"_cnn.txt"
    datasws=path1+"lay"+str(i+1)+"_sws.txt"
    mapname_cnn ="cnn_"+str(dep)+"km.eps"
    command_cnn= "bash ./plotSurfaceTomography.sh "+ datacnn1+" " + datacnn2 + " " +datasws+" "+ cptrange+" "+dep+" "+ mapname_cnn
    os.system(command_cnn)
    os.system("rm -rf  *.eps && mv *.png ./Figures")