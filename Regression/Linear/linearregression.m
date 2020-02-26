function [w, b] = linearregression(traindata, trainlabels)
    param = pinv((traindata.') * traindata) * (traindata.') * trainlabels;
    w = param(1);
    b = param(2);
end