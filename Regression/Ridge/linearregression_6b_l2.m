function [w, b] = linearregression_6b_l2(traindata, trainlabels, lambda)
% This implements ridge regression
    M = size(traindata, 1);
    I = [eye(8) zeros(8,1); zeros(1,9)];
    param = pinv((traindata.') * traindata + lambda * M * I) * (traindata.') * trainlabels;
    w = param(1:8,:);
    b = param(9);
end