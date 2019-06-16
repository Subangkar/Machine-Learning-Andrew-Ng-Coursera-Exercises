function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_set = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_set = [0.01 0.03 0.1 0.3 1 3 10 30];
% C = 0.3;
% sigma = 0.1;%0.03
err = zeros(size(C_set,2),size(sigma_set,2));%+size(sigma_set,2)
% for i=1:size(C_set,2)
%     for j=1:size(sigma_set,2)
%         model = svmTrain(X, y, C_set(i), @(x1, x2) gaussianKernel(x1, x2, sigma_set(j)));
%         predictions = svmPredict(model, Xval);
%         err(i,j) = mean(double(predictions ~= yval));
%     end
% end

[row,col] = find(err==min(err(:)));
C = C_set(1,row);
sigma = sigma_set(1,col);
%plot(C_set,err);

% plot(sigma_set,err);
C = 1;
sigma = 0.1;
% pause;
% =========================================================================

end
