%%https://sites.google.com/site/cs534aiwpi/projects/neural-network-project

%% Initialization
clear ; close all; clc


 
num_labels = 4;             
num_rows=548;
                          


%  dataset that contains 120x128 resolution images with labels sad:1,angry:2,neutral:3,happy:4.




load('features_emotion.mat'); % training data stored in arrays
X= features_emotion([1:547],:);
load('labels_emotion.mat');%output data for training set
m = size(X, 1);
rand_indices = randperm(m);
sel = X(rand_indices(1:25), :);

%% Display randomly selected 25 data points.
displayData(sel);

fprintf('\nTraining One-vs-All Logistic Regression...\n')
y=zeros(548,1);
lambda = 0.525;
for i=1:num_rows
[val,ind]=max(labels_emotion(i,:));
y(i)=ind;
end;
[all_theta] = oneVsAll(X, y([1:547],:), num_labels, lambda);
disp(size(all_theta));

%  Prediction Accuracy
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y([1:547],1))) * 100);
%Accuracy in new features
%prob = predictOneVsAll(all_theta,features_emotion(548,:));
%disp(size(prob));
%fprintf([' predicted probability of new features \n\n']);
%fprintf('\nNew Set Accuracy: %f\n', mean(double(prob == y(548,1))) * 100);


