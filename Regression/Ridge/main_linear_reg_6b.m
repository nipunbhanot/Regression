trainlabels_10_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\Train-subsets\y_train_10%.txt';
traindata_10_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\Train-subsets\X_train_10%.txt';
trainlabels_100_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\Train-subsets\y_train_100%.txt';
traindata_100_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\Train-subsets\X_train_100%.txt';
testlabels_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\y_test.txt';
testdata_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\X_test.txt';



[train_error_10percent, test_error_10percent, w_final_10, b_final_10] = cal_error_linear_reg_6b(trainlabels_10_path, traindata_10_path, testlabels_path, testdata_path);
[train_error_100percent, test_error_100percent, w_final_100, b_final_100] = cal_error_linear_reg_6b(trainlabels_100_path, traindata_100_path, testlabels_path, testdata_path);


fprintf('The training error obtained when training on the 10 percent data set is %f\n', train_error_10percent);
fprintf('The testing error obtained when training on the 10 percent data set is %f\n', test_error_10percent);

fprintf('The training error obtained when training on the full 100 percent data set is %f\n', train_error_100percent);
fprintf('The testing error obtained when training on the full 100 percent data set is %f\n', test_error_100percent);

fprintf('The weight vector from 10 percent training data is %f %f %f %f %f %f %f %f\n', w_final_10);
fprintf('The bias from 10 percent training data is %f\n', b_final_10);
fprintf('The weight vector from 100 percent training data is %f %f %f %f %f %f %f %f\n', w_final_100);
fprintf('The bias from 100 percent training data is %f\n', b_final_100);


