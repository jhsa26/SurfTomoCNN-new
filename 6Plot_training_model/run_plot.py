import numpy as np
import os
depth=np.array([0,5,10,15,20,30,40,60,80,100,120])
cpt=["1.3/2.0","3.1/3.6","3.4/3.7","3.4/3.8","3.5/3.9","3.6/4.2","3.8/4.6","4.1/4.7","4.1/4.7","4.1/4.8","4.1/4.8","4.1/4.8","4.1/4.7"]
#     0         3         10         15       20        30          40       3.6 60   80         100      120      150       180\n",
os.system("rm -rf Figures")
os.system("mkdir  Figures")
path="./layers_vs/"
for i in range(len(depth)):
    dep = str(depth[i])
    cptrange = cpt[i]
    datasws=path+"lay"+str(i+1)+".txt"
    mapname_sws ="sws_"+str(dep)+"km.eps"
    command_sws= "bash ./plotSurfaceTomography.sh "+ datasws + " "+ cptrange+" "+dep+" "+ mapname_sws
    os.system(command_sws)
    os.system("rm -rf  *.eps && mv *.png ./Figures")
