function [w, b] = LogisticRegressionL2(traindata, trainlabels, lambda)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data
    % lambda      - regularization parameter (positive real number)
        
    % OUTPUT
    % returns learnt model: w - n x 1 weight vector, b - bias term
    options = optimoptions('fminunc', 'MaxFunEvals', 2e6, 'MaxIter', 2e6);
    
    weight_initial = zeros(58, 1); %Initial
    weight = fminunc(@(w)(log_likelihood_l2_reg(traindata, trainlabels, w, lambda)), weight_initial, options);
    
    b = weight(58);
    w = weight(1:57);
    
end
