% This is the main code, which has the path for the data and labels.
% The plot function is called here.

trainlabels_main = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-4\Spambase\Train-subsets\y_train_';
traindata_main = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-4\Spambase\Train-subsets\X_train_';
train_path_filename = '%.txt';
testlabel_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-4\Spambase\y_test.txt';
testdata_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-4\Spambase\X_test.txt';
   
train_error = ones(10, 1);
test_error = ones(10, 1);

for i = 1:10
    [train_error(i), test_error(i)] = cal_error([trainlabels_main, num2str(i*10), train_path_filename], [traindata_main, num2str(i*10), train_path_filename], testlabel_path, testdata_path);
end

train_error_100percent = train_error(10);
test_error_100percent = test_error(10);

plot_error(train_error, test_error);

fprintf('The training error obtained when training on the full 100 percent data set is %f\n', train_error_100percent);
fprintf('The testing error obtained when training on the full 100 percent data set is %f\n', test_error_100percent);
