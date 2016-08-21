function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];


max_index=0;
for i=1:m 
[val,index]=max(sigmoid(X(i,:)*all_theta'));
p(i,1)=index;
disp(p(i,1))
end

end
