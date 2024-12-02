function A = esprit(spectrum,K)
N = length(spectrum);
Y = hankel(spectrum(1:end-K),spectrum(end-K:end));
Y1 = Y(:,1:end-1);
Y2 = Y(:,2:end);
l = eig(pinv(Y1)*Y2);A=l;
end

