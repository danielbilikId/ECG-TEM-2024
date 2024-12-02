function [phi,pulseHat] = genPulse(t,scale,supp,K)
% Generates a time-scaled cubic spline pulse
% along with its Fourier transform evaluated
% at harmonics of the fundamental
% 
% INPUT:  Time support, t
%         Time scale, scale
%         Signal support, supp
%         Number of pulses, K
%        
% OUTPUT: Pulse, phi
%         Fourier coefficients, pulseHat
%         
% Author: Abijith J Kamath
% kamath-abhijith.github.io
% abijithj@iisc.ac.in

    % Scale the time support
    t = t*scale; nt = length(t);
    phi = zeros(1,nt);
    
    % Evaluate cubic spline in closed form
    for i=1:nt
       if abs(t(i))<1
           phi(i) = (2/3)-abs(t(i))^2+0.5*abs(t(i))^3;
       elseif (abs(t(i))>=1) && (abs(t(i))<=2)
           phi(i) = (1/6)*(2-abs(t(i)))^3;
       else
           phi(i) = 0;
       end
    end
    
    % Evaluate finite Fourier coefficients in closed form
    w0 = 2*pi/supp;
    pulseHat = (1/(2*scale))*(sinc((-K:K)*w0/(2*pi*scale))).^4;
    pulseHat(K+1) = 2*pulseHat(K+1);
end

