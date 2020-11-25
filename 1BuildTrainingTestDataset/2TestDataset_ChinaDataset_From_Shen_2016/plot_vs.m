clear
close all





names = dir('./InputVsData/');

for i=3:length(names)
    vs=load(['./InputVsData/' names(i).name]) ;
    if isempty(vs)
        disp(' aaaaa')
    else
        plot(vs(1:300,2),vs(1:300,1),'-r','linewidth',1);
        hold on
    end
end

xlabel('Vs(km/s)')
ylabel('Depth(km)')
 
set(gca,'ydir','rev')

set(gca,'linewidth',2)
set(gca,'fontsize',18)
