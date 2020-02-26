function [train_error, test_error, w, b] = cal_error_linear_reg_6b_l2(trainlabelspath, traindatapath, testlabelspath, testdatapath, lambda)
% This function does the following:
%   1. Read data and labels. Augment data with a column vector of 1s.
%   2. Call linearregression and obtain w and b.
%   3. Perform Classification for both training and test data.
% It returns training error and test error.

    trainlabels = importdata(trainlabelspath);
    traindata = importdata(traindatapath);
    testlabels = importdata(testlabelspath);
    testdata = importdata(testdatapath);
    
    %Augmenting a column vector of 1s so that the bias term is included
    traindata_aug = [traindata ones(size(traindata, 1), 1)];
    
    [w, b] = linearregression_6b_l2(traindata_aug, trainlabels, lambda);
    
    predicted_y_train = traindata*w + b;
    
    predicted_y_test = testdata*w + b;
    
    %Computation of Error
    train_error = mean_squared_error(trainlabels, predicted_y_train);
    test_error = mean_squared_error(testlabels, predicted_y_test);
end