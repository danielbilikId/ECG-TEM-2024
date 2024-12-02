function smsKernel = genSmsKernel(t,order,supp,spOrder)
% Generates SMS Kernel with all coefficients set to unity
% 
% INPUT:  Time support, t
%         Kernel order, order
%         Kernel support, supp
%         Spline order, spOrder
%
% OUTPUT: Periodized kernel, smsKernel
%
% Author: Abijith J Kamath
% kamath-abhijith.github.io
% abijithj@iisc.ac.in
%
% For more information, check:
% ieeexplore.ieee.org/abstract/document/7997739/

    % Compute dependencies
    Nt = length(t); dt = t(2)-t(1);
    tMax = max(t); w = 2*pi/supp;
    
    % Generate base kernel
    smsSpline = genSpline(t,spOrder,supp);
    smsKernel = (cos(w*t'*(0:order)*supp)*...
        ones(order+1,1))'.*smsSpline*dt;

    % Periodise kernel
    oneSec = (Nt-1)/(2*tMax);
    for iPer = 1:floor(order/2)
    smsKernel = smsKernel + ...
        [zeros(1,iPer*oneSec) smsKernel(1:end-iPer*oneSec)] + ...
        [smsKernel(iPer*oneSec+1:end) zeros(1,iPer*oneSec)];
    end    
end