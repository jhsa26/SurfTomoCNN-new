function [rho,vp]=get_rho_vp_cf(vs)
pi = 3.141592653;
vp = 0.9409+2.094*vs-0.8206*vs.^2+0.2683*vs.^3-0.0251*vs.^4;
rho1 = (vp+2.4)/3.125;
rho2 = (7.55+ sqrt(57-10.56*(6.86-vp)))/5.28;

fai = (pi/2).*((1+tanh(0.5.*(vp-6.2)))/2.0);

rho = cos(fai).*cos(fai).*rho2 + sin(fai).*sin(fai).*rho1;