function [J, grad] = costFunction(theta, X, y)
J = 0;
grad = zeros(size(theta));
m = length(y);
par_res=[-y.*log(sigmoid(X*theta))]-[(1-y).*log(1-sigmoid(X*theta))];
J=(1/m)*sum(par_res);
grad=(1/m)*[sum((sigmoid(X*theta)-y).*X)];
grad=grad(:);

end
