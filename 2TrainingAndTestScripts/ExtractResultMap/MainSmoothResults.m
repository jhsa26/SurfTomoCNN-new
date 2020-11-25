clc
clear 
close all
vs_layer_dir = './layers_vs/';
allfiles=strsplit(ls(vs_layer_dir))';
nfiles=length(allfiles)-1;
for i=1:nfiles
    filename_prefix = allfiles{i}; 
    temp=load([vs_layer_dir    allfiles{i,1}]);
    vel = [temp,ones(length(temp),1)*0.1];
    profile = vel(:,1:2);
    vs = krig_interp(profile,vel);
    temp(:,3) = vs(:);
    save([vs_layer_dir    allfiles{i,1}],'temp','-ascii')
end