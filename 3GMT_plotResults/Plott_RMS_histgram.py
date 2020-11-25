#!/usr/bin/env python
# coding: utf-8

# In[2]:


# plot final rms
import matplotlib.pyplot as plt
import pickle,pprint
import numpy as np
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 24
sws_ph_gr_chi=np.loadtxt('./layers_vs_usa/rms_sws_save.txt')
cnn_ph_gr_chi1=np.loadtxt('./layers_vs_usa/rms_cnn_save.txt') 
cnn_ph_gr_chi2=np.loadtxt('./layers_vs_usa_tibet/rms_cnn_save.txt')     
plt.rc('font',family='Times New Roman')    
font = {
        'size': 24,
        }
bins= np.linspace(0,5, 50)    
fig,axes=plt.subplots(1,3,figsize=(18,6))
ax = axes[0] 
ax.hist(cnn_ph_gr_chi1[:,-1],bins,color="grey",alpha=0.8)

ax.text(2.,100,"Mean Misfit:\n   {:.2f}".format(np.mean(cnn_ph_gr_chi1[:,-1])))
ax.set_xlabel("$\chi^2$ misfit",fontsize=font['size'])
ax.set_ylabel("Frequency",fontsize=font['size'])

ax.tick_params(axis="both",labelsize=font['size'])



ax = axes[1] 
ax.hist(cnn_ph_gr_chi2[:,-1],bins,color="grey",alpha=0.8)
# ax.set_title("Misfit",fontdict=font)
ax.text(2.,130,"Mean Misfit:\n   {:.2f}".format(np.mean(cnn_ph_gr_chi2[:,-1])))
ax.set_xlabel("$\chi^2$ misfit ",fontsize=font['size'])
ax.set_ylabel("Frequency",fontsize=font['size'])

ax.tick_params(axis="both",labelsize=font['size'])



ax = axes[2] 
ax.hist(sws_ph_gr_chi[:,-1],bins,color="grey",alpha=0.8)
ax.text(2,160,"Mean Misfit:\n   {:.2f}".format(np.mean(sws_ph_gr_chi[:,-1])))
# ax.set_title("Misfit",fontdict=font)
ax.set_xlabel("$\chi^2$ misfit ",fontsize=font['size'])
ax.set_ylabel("Frequency",fontsize=font['size'])
 
ax.tick_params(axis="both",labelsize=font['size'])
plt.rcParams["font.family"] = "Times New Roman"
plt.tight_layout()
plt.savefig("chi-square_ph_gr.png",dpi=300,bbox_inches='tight')
plt.pause(1)


# In[ ]:




