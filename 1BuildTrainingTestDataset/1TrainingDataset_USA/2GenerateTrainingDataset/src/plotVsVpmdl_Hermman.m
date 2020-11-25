% plot Vs & Vp model from the input model for the calculation of dispersion
% curve using Hermman's codes

function plotVsVpmdl_Hermman(disp_gr,disp_1,disp_ph,disp_2,mdlfile, VslineChar, plotVpIndex, VplineChar)

% mdlfile = '1DSphmodel.mdl';
% VslineChar = 'k-';
% plotVpIndex = 1;  % = 1: plot Vp model, else: do not plot
% VplineChar = 'r-';
% example of usage: plotVsVpmdl_Hermman('1DSphmodel.mdl', 'k-', 1, 'r-');
%                   plotVsVpmdl_Hermman('1DSphmodel.mdl', 'k-', 0, 'r-');


fmdl = fopen(mdlfile, 'r');
for i = 1:12
    temp = fgetl(fmdl);
end

mdl = zeros(1, 10);
k = 0;

tempmdl = fscanf(fmdl, '%f', [10 inf]);
mdl = tempmdl';

nlayer = size(mdl,1);

newmdl = zeros(2*nlayer, 10);
newmdl(1, :) = mdl(1,:);
newmdl(1,1) = 0;
newmdl(2, :) = mdl(1,:);
for i = 2:nlayer
    newmdl(2*(i-1)+1, :) = mdl(i,:);
    newmdl(2*(i-1)+1,1) = sum(mdl(1:(i-1),1));
    newmdl(2*i, :) = mdl(i,:);
    newmdl(2*i, 1) = sum(mdl(1:i,1));
end
newmdl(2*nlayer, 1) = newmdl(2*nlayer-1, 1) + 10;
subplot(1,2,1)
plot(disp_gr(:,1),disp_gr(:,2),'-r','linewidth',2)
hold on
plot(disp_1(:,1),disp_1(:,2),'--r','linewidth',2)
plot(disp_ph(:,1),disp_ph(:,2),'-g','linewidth',2)
plot(disp_2(:,1),disp_2(:,2),'--g','linewidth',2)
legend({'disp-gr-syn','disp-gr-real','disp-ph-syn','disp-ph-real'})
subplot(1,2,2)
plot(newmdl(:,3), newmdl(:,1), VslineChar, 'LineWidth',1);
if plotVpIndex == 1
    hold on
    plot(newmdl(:,2), newmdl(:,1), VplineChar, 'LineWidth',2);
    title('Vs and Vp model','FontSize',12);
else
    title('Vs model','FontSize',12);
end
set(gca, 'YDir', 'reverse', 'FontSize',12)    
xlabel('Velocity (km/s)', 'FontSize',12);
ylabel('Depth (km)', 'FontSize', 12);
