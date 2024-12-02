function s = time_eval(ck,rk,tk,t,k)
    s = (ck(k)/pi)*rk(k)./(rk(k).^2+(t-tk(k)).^2); 
end