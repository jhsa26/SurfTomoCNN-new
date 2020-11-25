# coding: utf-8
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import pickle
disp_real_dir='./data/disp_pg_real/'
disp_sws_dir ='./data/disp_sws/'
disp_cnn_dir ='./data/disp_cnn_usa/'
disp_cnn_dir='./data/disp_cnn_usa_tibet/'
vs_sws_dir = './data/vs_sws_China/'
# vs_cnn_dir ='./data/vs_with_uncertainty_60x17_0.1_usa/'
vs_cnn_dir="./data/vs_with_uncertainty_60x17_0.1_usa_tibet/"
allfiles=os.listdir((disp_cnn_dir))
count = 0
cnn_ph_chi=[]
cnn_gr_chi=[]
sws_ph_chi = []
sws_gr_chi = []
cnn_ph_gr_chi = []
sws_ph_gr_chi = []
font={
    'size':18  
}
region = [0,5,150,0]
os.system("rm -rf Figs_usa && mkdir Figs_usa")

def fun_mean_square(v):
    v2 = v*v
    chi_square =np.sqrt(np.sum(v2)/len(v))
    return chi_square 
with open('./data/select.point','r') as f:
    chinaNames=f.read().splitlines()

lon_lat_save=['100_36','120_31','115_25','92_43','93_33','108_38','106_30','124_46','117_34'];
# for fi in allfiles:
for fi in chinaNames:
    temp=fi.split()
    lon,lat=fi.split()
    fi = lat+"_"+lon+'.txt'
    a=os.path.exists(disp_cnn_dir+fi)
    b=os.path.exists(disp_real_dir+fi)
    c=os.path.exists(disp_sws_dir+fi)
    if a and b and c:
        count =count+1
        
        lat,lon=fi.split('.txt')[0].split('_')
        lon_lat_select = '_'.join([lon.split('.')[0],lat.split('.')[0]])
        lat=float(lat);lon=float(lon)
        if lon_lat_select in lon_lat_save : #count % 50 ==0:
            cnn_disp = np.loadtxt(disp_cnn_dir+fi)
            real_disp = np.loadtxt(disp_real_dir+fi)
            sws_disp = np.loadtxt(disp_sws_dir+fi)
            vs_cnn = np.loadtxt(vs_cnn_dir+fi)
            vs_sws = np.loadtxt(vs_sws_dir+fi)
            periods = real_disp[:,0]
            ph_disp = real_disp[:,1];ph_un = real_disp[:,2]
            gr_disp = real_disp[:,3];gr_un = real_disp[:,4]
            # cnn chi-square
            chi_dv_ph=(cnn_disp[:,1]-ph_disp)/ph_un
            rms = fun_mean_square(chi_dv_ph)
            cnn_ph_chi.append([lon,lat,rms])

            chi_dv_gr=(cnn_disp[:,2]-gr_disp)/gr_un
            rms = fun_mean_square(chi_dv_gr)
            cnn_gr_chi.append([lon,lat,rms])
            # sws chi-square
            chi_dv_ph=(sws_disp[:,1]-ph_disp)/ph_un
            rms = fun_mean_square(chi_dv_ph)
            sws_ph_chi.append([lon,lat,rms])
            chi_dv_gr=(sws_disp[:,2]-gr_disp)/gr_un
            rms = fun_mean_square(chi_dv_gr)
            sws_gr_chi.append([lon,lat,rms])

            #cnn gr_ph stack
            ph_gr_cnn  = np.hstack((cnn_disp[:,1],cnn_disp[:,2]))
            ph_gr_disp = np.hstack((ph_disp,gr_disp))
            ph_gr_un   = np.hstack((ph_un,gr_un))
            chi_dv_ph_gr=(ph_gr_disp-ph_gr_cnn)/ph_gr_un
            rms = fun_mean_square(chi_dv_ph_gr)
            cnn_ph_gr_chi.append([lon,lat,rms])
            #sws gr_ph stack
            ph_gr_sws  = np.hstack((sws_disp[:,1],sws_disp[:,2]))
            ph_gr_disp = np.hstack((ph_disp,gr_disp))
            ph_gr_un   = np.hstack((ph_un,gr_un))
            chi_dv_ph_gr=(ph_gr_disp-ph_gr_sws)/ph_gr_un
            rms = fun_mean_square(chi_dv_ph_gr)
            sws_ph_gr_chi.append([lon,lat,rms])
        
            print('count ',count,fi)
            fig,axes=plt.subplots(1,3,figsize=(15,5))
            ax = axes[0]
            ax.errorbar(periods[:],ph_disp[:],yerr=ph_un,fmt='-ro',linewidth=2,label="Observed", capsize=4 )
            ax.plot(periods[:],cnn_disp[:,1],"-g",linewidth=2,label="CNN")
            ax.plot(periods[:],sws_disp[:,1],"-b",linewidth=2,label="Shen et al. (2016)")
            ax.set_title("Phase velocity",fontsize=font['size'])
            ax.set_xlabel("T(s)",fontsize=font['size'])
            ax.set_ylabel("Velocity(km/s)",fontsize=font['size'])
            ax.tick_params(axis="both",labelsize=font['size'])
            ax.legend(loc="lower right",fontsize = font["size"]-2)
            
            ax = axes[1]
            ax.errorbar(periods[:],gr_disp[:],yerr=gr_un,fmt="-or",linewidth=2,label="Observed", capsize=4 )
            ax.plot(cnn_disp[:,0],cnn_disp[:,2],'-g',linewidth=2,label="CNN")
            ax.plot(periods[:],sws_disp[:,2],"-b",linewidth=2,label="Shen et al. (2016)")
            

            ax.set_title("Group velocity",fontsize=font['size'])
            ax.set_xlabel("T(s)",fontsize=font['size'])
            ax.set_ylabel("Velocity(km/s)",fontsize=font['size'])
            ax.tick_params(axis="both",labelsize=font['size'])
            ax.legend(loc="lower right",fontsize=font["size"]-2)
            
            ax=axes[2]
            mean=vs_cnn[:,1]
            vs_un=vs_cnn[:,2]
            deterministic=vs_cnn[:,3]
            upperbound=np.max(vs_cnn[:,4:],axis=1)
            lowerbound=np.min(vs_cnn[:,4:],axis=1)
            ax.fill_betweenx(vs_cnn[:,0],lowerbound,upperbound,alpha=0.3,color='gray')
            
            lns1=ax.plot(vs_cnn[:,3],vs_cnn[:,0],"-r",linewidth=2,label="CNN")
            lns2=ax.plot(vs_sws[:,1],vs_sws[:,0],"-",color="yellow",linewidth=2,label="Shen et al. (2016)")
            
            ax2=ax.twiny()
            lns3=ax2.plot(vs_cnn[:,2],vs_cnn[:,0],"-",color="gray",linewidth=2,label="Uncertainty")
            ax2.set_xlabel('Uncertainty (km/s)', color="black") 
            ax2.xaxis.label.set_color('gray')
            ax.axis(region)
            ax.set_ylabel("Depth(km)",fontsize=font['size'])
            ax.set_xlabel("Shear wave velocity (km/s)",fontsize=font['size'])
#             ax.set_title("1-D velocity model",fontsize=font['size'])
            ax.tick_params(axis="both",labelsize=font['size'])
            lns = lns1+lns2+lns3
            labs = [l.get_label() for l in lns]
            ax.legend(lns,labs,loc = "lower left",fontsize=font["size"]-2)
            plt.tight_layout()
            plt.savefig("./Figs_usa/"+fi+".png",dpi=300,bbox_inches="tight")
            plt.pause(0.1)
            fig.clf()
