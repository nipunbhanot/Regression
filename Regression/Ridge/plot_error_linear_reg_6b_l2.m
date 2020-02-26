function plot_error_linear_reg_6b_l2(train_error, test_error, crossval_test_error_avg)
% This function plots the test curve.
iter = (1:6);
figure;
plot(iter, train_error);
hold on;
plot(iter, test_error);
hold on;
plot(iter, crossval_test_error_avg); 
xlabel('Regularization Parameter (Lambda = 0.1, 1, 10, 100, 500, 1000)');
ylabel('Error');
legend('Training Error', 'Test Error', 'Cross-Validation Error');
title('Learning Curve')
end

