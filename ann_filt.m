function [freq_est]=ann_filt(y,K)
% High-Resolution Spectral Estimation (HRSE) from 
% sequence in Sum of Weighted Complex Exponential (SWCE) form
% 
% Frequencies are estimated using the roots of an
% annihilating filter. Solution to the null space is
% computed using Eckart-Young Theorem.
%
% INPUT:  Sequence in SWCE form, t
%         Model Order, K
% OUTPUT: Estimated frequencies, freq_est
%
% Written by: Kfir Gedalyahu (kfir.gedal@gmail.com)
%
% For more information, check: sparse sampling of signal innovations

% Preprocessing
y = y(:);
N=length(y);
M=floor(N/2);

% Error checks
if N<2*K || ~mod(N,2)
    error('Number of samples must be greater than 2K and odd');
end

% Construct the convolution matrix
A=zeros(2*M-K+1,K+1);
for i=0:2*M-K
    A(i+1,:)=y((-M+K:-1:-M)+ceil(N/2)+i);
end

% Solution to the null space
[~,~,V]=svd(A,0);    
h=V(:,K+1);

% Estimating frequencies using roots of the filter
freq_est = angle(roots(h));
freq_est(freq_est>0) = freq_est(freq_est>0)-2*pi;
end
