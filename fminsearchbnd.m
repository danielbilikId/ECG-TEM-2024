function [x,fval,exitflag,output] = fminsearchbnd(fun,x0,LB,UB,options)
% fminsearchbnd: FMINSEARCH, but with bound constraints by transformation
% [x,fval,exitflag,output] = fminsearchbnd(fun,x0,LB,UB,options)
% 
% This function extends the functionality of `fminsearch` by adding the
% capability to enforce bound constraints on the variables.

% Argument validation
if nargin < 3 || isempty(LB), LB = -inf(size(x0)); end
if nargin < 4 || isempty(UB), UB = inf(size(x0)); end
if nargin < 5, options = optimset(); end

% Transform to work with bound constraints
x0_trans = transformX(x0, LB, UB);
fun_trans = @(x) fun(invTransformX(x, LB, UB));

% Call fminsearch on the transformed problem
[x_trans, fval, exitflag, output] = fminsearch(fun_trans, x0_trans, options);

% Transform solution back to the original space
x = invTransformX(x_trans, LB, UB);

end

% Transformation function (maps x to transformed space)
function xt = transformX(x, LB, UB)
xt = log((x - LB) ./ (UB - x));
end

% Inverse transformation function (maps transformed x back to original space)
function x = invTransformX(xt, LB, UB)
x = LB + (UB - LB) ./ (1 + exp(-xt));
end
