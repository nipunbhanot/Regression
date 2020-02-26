function [w, b] = linearregression_6b(traindata, trainlabels)
    param = pinv((traindata.') * traindata) * (traindata.') * trainlabels;
    w = param(1:8,:);
    b = param(9);
end