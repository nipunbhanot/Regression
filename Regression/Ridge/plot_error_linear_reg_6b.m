function plot_error_linear_reg_6b(train_error, test_error)
% This function plots the test curve.
iter = 1:10;
figure;
plot(iter*10, train_error);
hold on;
plot(iter*10, test_error);
xlabel('Percentage of Training Data Used');
ylabel('Error');
legend('Training Error', 'Test Error');
title('Test Curve')
end

