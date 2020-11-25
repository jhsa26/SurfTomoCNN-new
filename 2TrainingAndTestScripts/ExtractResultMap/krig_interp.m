
function vs = krig_interp(profile,vel)

    x = profile(:,1);       y = profile(:,2);
    M = size(profile,1);    vs = zeros(M,1);

    for i = 1:M

        d = ( (x(i)-vel(:,1)).^2  + (y(i)-vel(:,2)).^2 ).^0.5 ;
        d(d>1) = 0;
        d = d+1;
        d(d==1) = 0;
        tp = d.*vel(:,4);
        id = find(tp);
        tp = tp(tp>0);
        vs(i) = sum( vel(id,3)./tp )/sum( 1./tp );
        
    end

end
