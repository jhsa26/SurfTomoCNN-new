from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from mpl_toolkits.basemap import Basemap

periods = np.array([8,10,12,14,16,18,20,22,24,26,28,30,32,35,40,45,50])
depth = np.array([0,3,10,15,20,30,40,60,80,100,120,150])
lay1_sws=np.loadtxt('./layers_vs_usa/lay1_sws.txt')
lay2_sws=np.loadtxt('./layers_vs_usa/lay2_sws.txt')
lay3_sws=np.loadtxt('./layers_vs_usa/lay3_sws.txt')
lay4_sws=np.loadtxt('./layers_vs_usa/lay4_sws.txt')
lay5_sws=np.loadtxt('./layers_vs_usa/lay5_sws.txt')
lay6_sws=np.loadtxt('./layers_vs_usa/lay6_sws.txt')
lay7_sws=np.loadtxt('./layers_vs_usa/lay7_sws.txt')
lay8_sws=np.loadtxt('./layers_vs_usa/lay8_sws.txt')
lay9_sws=np.loadtxt('./layers_vs_usa/lay9_sws.txt')
lay10_sws=np.loadtxt('./layers_vs_usa/lay10_sws.txt')
lay11_sws=np.loadtxt('./layers_vs_usa/lay11_sws.txt')


lay1_cnn_usa=np.loadtxt('./layers_vs_usa/lay1_cnn.txt')
lay2_cnn_usa=np.loadtxt('./layers_vs_usa/lay2_cnn.txt')
lay3_cnn_usa=np.loadtxt('./layers_vs_usa/lay3_cnn.txt')
lay4_cnn_usa=np.loadtxt('./layers_vs_usa/lay4_cnn.txt')
lay5_cnn_usa=np.loadtxt('./layers_vs_usa/lay5_cnn.txt')
lay6_cnn_usa=np.loadtxt('./layers_vs_usa/lay6_cnn.txt')
lay7_cnn_usa=np.loadtxt('./layers_vs_usa/lay7_cnn.txt')
lay8_cnn_usa=np.loadtxt('./layers_vs_usa/lay8_cnn.txt')
lay9_cnn_usa=np.loadtxt('./layers_vs_usa/lay9_cnn.txt')
lay10_cnn_usa=np.loadtxt('./layers_vs_usa/lay10_cnn.txt')
lay11_cnn_usa=np.loadtxt('./layers_vs_usa/lay11_cnn.txt')


lay1_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay1_cnn.txt')
lay2_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay2_cnn.txt')
lay3_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay3_cnn.txt')
lay4_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay4_cnn.txt')
lay5_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay5_cnn.txt')
lay6_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay6_cnn.txt')
lay7_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay7_cnn.txt')
lay8_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay8_cnn.txt')
lay9_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay9_cnn.txt')
lay10_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay10_cnn.txt')
lay11_cnn_usa_tibet=np.loadtxt('./layers_vs_usa_tibet/lay11_cnn.txt')

all_lyers_sws = [lay1_sws,lay2_sws,lay3_sws,lay4_sws,lay5_sws,lay6_sws,
                 lay7_sws,lay8_sws,lay9_sws,lay10_sws,lay11_sws];
all_lyers_cnn_usa = [lay1_cnn_usa,lay2_cnn_usa,lay3_cnn_usa,lay4_cnn_usa,lay5_cnn_usa,lay6_cnn_usa,
                     lay7_cnn_usa,lay8_cnn_usa,lay9_cnn_usa,lay10_cnn_usa,lay11_cnn_usa];
all_lyers_cnn_usa_tibet = [lay1_cnn_usa_tibet,lay2_cnn_usa_tibet,lay3_cnn_usa_tibet,lay4_cnn_usa_tibet,
                           lay5_cnn_usa_tibet,lay6_cnn_usa_tibet,lay7_cnn_usa_tibet,lay8_cnn_usa_tibet,
                           lay9_cnn_usa_tibet,lay10_cnn_usa_tibet,lay11_cnn_usa_tibet];
count=0
cm=[[2.0,3.6],[2.5,3.5],[3.1,3.7],[3.2,3.8],[3.2,3.9],[3.2,4.2],[3.4,4.6],[3.7,4.6],[4.15,4.7],[4.2,4.8],[4.05,4.75],[4.1,4.8],[4.1,4.7]]
#     0         3         10         15       20        30          40       3.6 60   80         100      120      150       180

pad=80
scale=0.3
fontsize=18
for ilay in range(len(all_lyers_cnn_usa)):
    count =count+1
    fig=plt.figure(count,figsize=(18,10))
    ax = plt.subplot(1,3,1)
    m = Basemap(projection='mill', llcrnrlon=73, llcrnrlat=18, urcrnrlon=135,
            urcrnrlat=53)
    m.drawcountries(linewidth=1.5)
    m.drawcoastlines(linewidth=1.5)
    m.drawlsmask(land_color="0.8", ocean_color="w", lsmask=None, lsmask_lons=None, 
                 lsmask_lats=None, lakes=True, grid=1.25)
    layer_sws = all_lyers_sws[ilay] 
    x_sws,y_sws = m(layer_sws[:,0],layer_sws[:,1])
    z_sws = layer_sws[:,2]
    sc=plt.scatter(x_sws,y_sws,c=z_sws,s=10,cmap='jet_r',marker='o')
    ax.set_title('SWS: Depth= '+str(depth[count-1])+'km',fontsize=fontsize)
    # draw parallels
    m.drawparallels(np.arange(10,90,10),labels=[1,0,0,1],fontsize=fontsize)
    # draw meridians
    m.drawmeridians(np.arange(-180,180,15),labels=[1,1,0,1],fontsize=fontsize)
    plt.clim(cm[count-1])
    
    ax = plt.subplot(1,3,2) 
    m = Basemap(projection='mill', llcrnrlon=73, llcrnrlat=18, urcrnrlon=135,
            urcrnrlat=53)
    m.drawcountries(linewidth=1.5)
    m.drawcoastlines(linewidth=1.5)
    m.drawlsmask(land_color="0.8", ocean_color="w", lsmask=None, lsmask_lons=None, 
                 lsmask_lats=None, lakes=True, grid=1.25)
    layer_cnn = all_lyers_cnn_usa[ilay]
    x_cnn,y_cnn = m(layer_cnn[:,0],layer_cnn[:,1])
    z_cnn = layer_cnn[:,2]
    sc=plt.scatter(x_cnn,y_cnn,c=z_cnn,s=10,cmap='jet_r',marker='o')
    ax.set_title('USA: Depth= '+str(depth[count-1])+'km',fontsize=fontsize)
    # draw parallels
    m.drawparallels(np.arange(10,90,10),labels=[0,0,0,0],fontsize=fontsize)
    # draw meridians
    m.drawmeridians(np.arange(-180,180,15),labels=[1,1,0,1],fontsize=fontsize)
    plt.clim(cm[count-1])
    
    ax = plt.subplot(1,3,3) 
    m = Basemap(projection='mill', llcrnrlon=73, llcrnrlat=18, urcrnrlon=135,
            urcrnrlat=53)
    m.drawcountries(linewidth=1.5)
    m.drawcoastlines(linewidth=1.5)
    m.drawlsmask(land_color="0.8", ocean_color="w", lsmask=None, lsmask_lons=None, 
                 lsmask_lats=None, lakes=True, grid=1.25)
    layer_cnn = all_lyers_cnn_usa_tibet[ilay]
    x_cnn,y_cnn = m(layer_cnn[:,0],layer_cnn[:,1])
    z_cnn = layer_cnn[:,2]
    sc=plt.scatter(x_cnn,y_cnn,c=z_cnn,s=10,cmap='jet_r',marker='o')
    ax.set_title('USA+Tibet: Depth= '+str(depth[count-1])+'km',fontsize=fontsize)
    # draw parallels
    m.drawparallels(np.arange(10,90,10),labels=[0,0,0,0],fontsize=fontsize)
    # draw meridians
    m.drawmeridians(np.arange(-180,180,15),labels=[1,1,0,1],fontsize=fontsize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb=plt.colorbar(sc, cax=cax)
#     cb.set_label(label="Vs(km/s)",size=fontsize, labelpad=-20)
    cb.ax.tick_params(axis='both', which='major', labelsize=fontsize)
    cb.ax.set_title('Vs(km/s)',fontsize=fontsize-2,pad=10)
    plt.clim(cm[count-1])
    plt.tight_layout()
    plt.savefig( './Figs_vs/data_setdif_sws_cnn-'+str(depth[count-1])+'.jpg',bbox_inches='tight',dpi=300) 
    plt.pause(0.1)
    fig.clear()
