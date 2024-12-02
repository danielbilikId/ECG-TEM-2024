function phi = genSpline(t,n,scale)
% Full vector evaluation of an nth order spline
% 
% INPUT:  Support vector, t
%         Spline order, n
%         Scale, scale
%
% OUTPUT: Spline kernel vector, phi
%
% Author: Abijith J Kamath
% kamath-abhijith@github.io
%
% For more information, check:
% "Splines: A perfect fit for signal
% and image processing" - Michael Unser
    
    % Rescale support
    t = t*scale;
    
    % Define first order spline
    zerosp = zeros(1,length(t));
    zerosp(abs(t)<0.5) = 1;

    % Compute nth order spline
    phi = zerosp;
    for i=1:n
        phi = conv(phi,zerosp,'same'); 
    end
    phi = phi/(max(phi));
end

% T0SMS = 32; w0SMS = 2*pi/T0SMS; K = 11;
% smsSpline = gen_spline_kernel(tLPF,2,T0SMS)/11;
% smsKernel = (cos(w0SMS*tLPF'*(0:K))*ones(K+1,1))'.*smsSpline;
% 
% y = conv(x,smsKernel,'same')/norm(smsKernel)^2;