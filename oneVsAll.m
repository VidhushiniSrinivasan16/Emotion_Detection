function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);
%alpha=1;
%num_itr=2000;

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];



options = optimset('GradObj', 'on', 'MaxIter', 150);
initial_theta=zeros(n+1,1);
i=1
for c=1:num_labels
[theta,J,exitFlag]=fmincg(@(t)(costFunction(t,X,(y==c))),initial_theta,options);
%[theta,J]=gradientDescentMulti(X,y,initial_theta,alpha,num_itr);
%disp(size(theta));
theta=theta(:);
%disp(size(theta'));
%disp(size(all_theta));

all_theta(i,:)=theta';
i=i+1;
end

end
