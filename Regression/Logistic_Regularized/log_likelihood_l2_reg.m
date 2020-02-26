function log_lik = log_likelihood_l2_reg(traindata, trainlabels, w, lambda)
% To compute the log_likelihood term which is used for optimization in
% LogisticRegression.m
    log_lik = sum(log(1 + exp(-trainlabels.*traindata*w)))/length(trainlabels) + lambda*sum(w.^2);
end