function plot_error_l2_reg(train_error, test_error, crossval_test_error_avg)
% This function plots the test curve.
iter = -7:0;
figure;
plot(iter, train_error);
hold on;
plot(iter, test_error);
hold on;
plot(iter, crossval_test_error_avg); 
xlabel('Regularization Parameter (Lambda)');
ylabel('Error');
legend('Training Error', 'Test Error', 'Cross-Validation Error');
title('Learning Curve')
end

