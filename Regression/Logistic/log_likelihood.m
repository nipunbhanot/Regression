function log_lik = log_likelihood(traindata, trainlabels, w)
% To compute the log_likelihood term which is used for optimization in
% LogisticRegression.m
    log_lik = sum(log(1 + exp(-trainlabels.*traindata*w)))/length(trainlabels);
end