function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
n=length(theta);

par_res=[-y.*log(sigmoid(X*theta))]-[(1-y).*log(1-sigmoid(X*theta))];
J=(1/m)*sum(par_res)+(lambda/(2*m)).*sum(theta([2:n]).^2);
grad(1)=(1/m)*[sum((sigmoid(X*theta)-y).*X(:,1))];
grad([2:n])=[(1/m)*[sum((sigmoid(X*theta)-y).*X(:,[2:n]))]]'+(lambda/m).*theta([2:n]);


grad = grad(:);

end
