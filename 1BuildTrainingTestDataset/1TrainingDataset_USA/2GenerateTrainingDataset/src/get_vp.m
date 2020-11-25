function tp = get_vp(ts,flag,vpvs)
%vpvs = 0

    if(flag==1)     % (Brocher, 2005); crust
        tp = 0.9409 + 2.0947*ts - 0.8206*ts*ts + ...
                  0.2683*ts*ts*ts -0.0251*ts*ts*ts*ts;
        tp = ts*(tp/ts + vpvs);
        
    elseif(flag==2) % AK135, 120km, vpvs=1.789; mantle 
        tp = ts*1.789;
        
    elseif(flag==3) % vpvs = vpvs
        tp = ts*vpvs;
        
    elseif(flag==4)
        tp = ts*(1.373+2.022/ts);
        
    else
        tp = ts*1.75;
    end

end
