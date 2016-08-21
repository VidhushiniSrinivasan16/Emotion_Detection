function J = computeCostMulti(X, y, theta)
m = length(y); % number of training examples
J = 0;
result1=(X*theta)-y;
sqrError=result1.^2;
J=1/(2*m) * sum(sum(sqrError));
end
