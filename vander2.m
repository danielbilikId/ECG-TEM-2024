function A = vander2(v,k)
%VANDER2 Vandermonde matrix.
%   A = VANDER2(V) returns the Vandermonde matrix whose columns
%   are powers of the vector V, that is A(:,j) = v^(j) 
%   for all j in [0 k-1]. If k is omitted, k=length(v).
%
%   Extends the built in MATLAB vander() function. 
%   Uses the more common definition of the Vandermonde matrix and adds the 
%   ability to create non-square matrices.
%
%   Class support for input V:
%      float: double, single
%
%   Adapted from the original MATLAB vander() function
if nargin==1
    k = length(v);
end
A = bsxfun(@power,v(:),0:k-1);
