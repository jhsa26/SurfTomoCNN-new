function rho = get_rho(tp,ts,flag,drho)
% drho = 0

    if( flag==1 )   % Brocher, 2005, crust
        trho = 1.22679 + 1.53201*ts -0.83668*ts*ts + ...
                0.20673*ts*ts*ts -0.01656*ts*ts*ts*ts;
       
    elseif( flag==2 ) % Brocher, 2005, mantle
        trho = 3.42+0.01*100*(ts-4.5)/4.5;
        
    elseif( flag==3 ) 
        trho = 1.05;
    
    elseif( flag==4 )
        trho = 2.7;

    elseif( flag==5 )
        trho = 3.38;

    elseif( flag==6 )
        trho = 0.31 * pow (tp*1000., 0.25);
    else
        trho = 1.22679 + 1.53201*ts -0.83668*ts*ts + ...
                0.20673*ts*ts*ts -0.01656*ts*ts*ts*ts;        
    end
    
    rho = trho + drho;    
    if(rho<0)  
        rho = 0;   
    end
    
end
