function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

result1=((X*theta)-y).*X;
delta=1/m * sum(result1);
theta=theta-(alpha*delta');

% Saving the cost J in every iteration    
J_history(iter) = costFunction(theta,X, y);

end

end
