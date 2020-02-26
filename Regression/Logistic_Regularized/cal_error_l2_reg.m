function [train_error, test_error] = cal_error_l2_reg(trainlabelspath, traindatapath, testlabelspath, testdatapath, lambda)
% This function does the following:
%   1. Read data and labels. Augment data with a column vector of 1s.
%   2. Call LogisticRegression and obtain w and b.
%   3. Compute Class Probabilities for training and test data.
%   4. Perform Classification for both training and test data.
% It returns training error and test error.

    trainlabels = importdata(trainlabelspath);
    traindata = importdata(traindatapath);
    testlabels = importdata(testlabelspath);
    testdata = importdata(testdatapath);
    
    %Augmenting a column vector of 1s so that the bias term is included
    traindata = [traindata ones(size(traindata, 1), 1)];
    testdata = [testdata ones(size(testdata, 1), 1)];
    
    [w, b] = LogisticRegressionL2(traindata, trainlabels, lambda);
    
    %Class probabilities for training samples
    prob = 1./(1 + exp(-traindata*[w; b]));
    
    %Classification
    temp = prob;
    for i = 1:size(temp,1)
        if prob(i) >= 0.5
            temp(i) = 1;
        else
            temp(i) = -1;
        end
    end
    predicted_y_train = temp;

    %Class probabilities for test samples
    prob_test = 1./(1 + exp(-testdata*[w; b]));
    
    %Classification
    temp = prob_test;
    for i = 1:size(temp,1)
        if prob_test(i) >= 0.5
            temp(i) = 1;
        else
            temp(i) = -1;
        end
    end
    predicted_y_test = temp;
    
    %Computation of Error
    train_error = classn_error(predicted_y_train, trainlabels);
    test_error = classn_error(predicted_y_test, testlabels);
end