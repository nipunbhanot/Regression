trainlabels_main = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-1\Train-subsets\y_train_';
traindata_main = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-1\Train-subsets\X_train_';
train_path_filename = '%.txt';
testlabel_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-1\y_test.txt';
testdata_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-1\X_test.txt';


train_error = ones(10, 1);
test_error = ones(10, 1);

for i = 1:10
    [train_error(i), test_error(i), w(i), b(i)] = cal_error_linear_reg([trainlabels_main, num2str(i*10), train_path_filename], [traindata_main, num2str(i*10), train_path_filename], testlabel_path, testdata_path);
end

train_error_100percent = train_error(10);
test_error_100percent = test_error(10);

plot_error_linear_reg(train_error, test_error);

fprintf('The training error obtained when training on the full 100 percent data set is %f\n', train_error_100percent);
fprintf('The testing error obtained when training on the full 100 percent data set is %f\n', test_error_100percent);

w_final = w(10);
b_final = b(10);

fprintf('The weight vector from the full training data is %f\n', w_final);
fprintf('The bias term from the full training data is %f\n', b_final);


x_model = importdata('C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-1\X_test.txt');
y_true = importdata('C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-1\y_test.txt');
y_pred = x_model*w_final + b_final;

figure;
scatter(x_model, y_true);
hold on;
plot(x_model, y_pred);
xlabel('Input Instance');
ylabel('Value of y');
legend('Original y', 'Learned Linear Function');
title('Learned Linear Function and Scatter Plot of Test Data')

