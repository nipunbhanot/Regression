function [w, b] = LogisticRegression(traindata, trainlabels)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data    
    
    % OUTPUT
    % returns learnt model: w - n x 1 weight vector, b - bias term
    
    % Fill in your code here    
    % Consider using fminunc MATLAB function for solving the logistic regression optimization problem.
    
    options = optimoptions('fminunc', 'MaxFunEvals', 2e6, 'MaxIter', 2e6);
    
    weight_initial = zeros(58, 1); %Initial
    weight = fminunc(@(w)(log_likelihood(traindata, trainlabels, w)), weight_initial, options);
    
    b = weight(58);
    w = weight(1:57);
    
end