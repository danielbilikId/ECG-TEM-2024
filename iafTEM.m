function [tnIdx,y_out] = iafTEM(x,dt,b,d,k)
% Integrate and Fire Time Encoding Machine
% 
% 
% INPUT:  Signal, x
%         Signal resolution, dt
%         Bias, b
%         Threshold, d
%         Integrator accelerator, k
%
% OUTPUT: Time stamp indices, tnIdx
%         Integrator output, t_out
%
% Author: Abijith J Kamath
% kamath-abhijith.github.io
% abijithj@iisc.ac.in

    % Preprocessing
    Nx = length(x); j = 1;

    % Rectangular approx.
    y = 0;
    compute_y = @(y,i) y + dt*(b+x(i))/k;

    % Search
    for i=1:Nx
        
      % Compute integrator output  
      y = compute_y(y,i);
      y_out(i) = y;
      
      % Threshold firing
      if y >= d
        tnIdx(j) = i;
        j = j + 1;
        y = y - d;
      end
    end
end