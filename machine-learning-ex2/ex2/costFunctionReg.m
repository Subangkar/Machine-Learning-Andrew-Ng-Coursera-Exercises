function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = @(X,t) sigmoid(t'*X);% X has to be column vectors

cost = @(X,y,theta) (-y.*log(h(X,theta))-(1-y).*log(1-h(X,theta)));

J = (1/(m))*sum(cost(X',y',theta));

delta = h(X',theta)-y';% X,y has to be column vectors

grad = (1/m)*(delta*X)';

regTheta = theta;
regTheta(1)=0;
regJ = (1/(2*m))*(lambda)*sum(regTheta.^2);
regGrad = (lambda/m)*(regTheta);

%size(J)
J = J + regJ; 
grad = grad + regGrad;

% =============================================================

end
