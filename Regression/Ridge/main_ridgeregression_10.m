trainlabels_10_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\Train-subsets\y_train_10%.txt';
traindata_10_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\Train-subsets\X_train_10%.txt';
trainlabels_100_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\Train-subsets\y_train_100%.txt';
traindata_100_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\Train-subsets\X_train_100%.txt';
testlabels_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\y_test.txt';
testdata_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\X_test.txt';

%Importing Cross Validation data
cross_val_path_root = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-6\Data-set-2\Cross-validation\Fold';
cross_val_path_data = '\X_10%.txt';
cross_val_path_labels = '\y_10%.txt';

crossvallabels_path1 = [cross_val_path_root, num2str(1), cross_val_path_labels];
crossvaldata_path1 = [cross_val_path_root, num2str(1), cross_val_path_data];
crossvallabels_path2 = [cross_val_path_root, num2str(2), cross_val_path_labels];
crossvaldata_path2 = [cross_val_path_root, num2str(2), cross_val_path_data];
crossvallabels_path3 = [cross_val_path_root, num2str(3), cross_val_path_labels];
crossvaldata_path3 = [cross_val_path_root, num2str(3), cross_val_path_data];
crossvallabels_path4 = [cross_val_path_root, num2str(4), cross_val_path_labels];
crossvaldata_path4 = [cross_val_path_root, num2str(4), cross_val_path_data];
crossvallabels_path5 = [cross_val_path_root, num2str(5), cross_val_path_labels];
crossvaldata_path5 = [cross_val_path_root, num2str(5), cross_val_path_data];


crossvallabels1 = importdata(crossvallabels_path1);
crossvallabels2 = importdata(crossvallabels_path2);
crossvallabels3 = importdata(crossvallabels_path3);
crossvallabels4 = importdata(crossvallabels_path4);
crossvallabels5 = importdata(crossvallabels_path5);
crossvaldata1 = importdata(crossvaldata_path1);
crossvaldata2 = importdata(crossvaldata_path2);
crossvaldata3 = importdata(crossvaldata_path3);
crossvaldata4 = importdata(crossvaldata_path4);
crossvaldata5 = importdata(crossvaldata_path5);


crossvallabels_train1 = [crossvallabels1; crossvallabels2; crossvallabels3; crossvallabels4];
crossvallabels_train2 = [crossvallabels2; crossvallabels3; crossvallabels4; crossvallabels5];
crossvallabels_train3 = [crossvallabels3; crossvallabels4; crossvallabels5; crossvallabels1];
crossvallabels_train4 = [crossvallabels4; crossvallabels5; crossvallabels1; crossvallabels2];
crossvallabels_train5 = [crossvallabels5; crossvallabels1; crossvallabels2; crossvallabels3];
crossvaldata_train1 = [crossvaldata1; crossvaldata2; crossvaldata3; crossvaldata4];
crossvaldata_train2 = [crossvaldata2; crossvaldata3; crossvaldata4; crossvaldata5];
crossvaldata_train3 = [crossvaldata3; crossvaldata4; crossvaldata5; crossvaldata1];
crossvaldata_train4 = [crossvaldata4; crossvaldata5; crossvaldata1; crossvaldata2];
crossvaldata_train5 = [crossvaldata5; crossvaldata1; crossvaldata2; crossvaldata3];
crossvallabels_test1 = crossvallabels5;
crossvallabels_test2 = crossvallabels1;
crossvallabels_test3 = crossvallabels2;
crossvallabels_test4 = crossvallabels3;
crossvallabels_test5 = crossvallabels4;
crossvaldata_test1 = crossvaldata5;
crossvaldata_test2 = crossvaldata1;
crossvaldata_test3 = crossvaldata2;
crossvaldata_test4 = crossvaldata3;
crossvaldata_test5 = crossvaldata4;


train_error = ones(6, 1);
test_error = ones(6, 1);

reg_param = [0.1 1 10 100 500 1000];

for i = 1:6
    [train_error(i), test_error(i)] = cal_error_linear_reg_6b_l2(trainlabels_10_path, traindata_10_path, testlabels_path, testdata_path, reg_param(i));
end

% 5-fold cross validation
crossval_train_error_avg = ones(6, 1);
crossval_test_error_avg = ones(6, 1);
for i = 1:6
    [crossval_train_error_1, crossval_test_error_1] = cal_error_linear_reg_6b_l2_crossval(crossvallabels_train1, crossvaldata_train1, crossvallabels_test1, crossvaldata_test1, reg_param(i));
    [crossval_train_error_2, crossval_test_error_2] = cal_error_linear_reg_6b_l2_crossval(crossvallabels_train2, crossvaldata_train2, crossvallabels_test2, crossvaldata_test2, reg_param(i));
    [crossval_train_error_3, crossval_test_error_3] = cal_error_linear_reg_6b_l2_crossval(crossvallabels_train3, crossvaldata_train3, crossvallabels_test3, crossvaldata_test3, reg_param(i));
    [crossval_train_error_4, crossval_test_error_4] = cal_error_linear_reg_6b_l2_crossval(crossvallabels_train4, crossvaldata_train4, crossvallabels_test4, crossvaldata_test4, reg_param(i));
    [crossval_train_error_5, crossval_test_error_5] = cal_error_linear_reg_6b_l2_crossval(crossvallabels_train5, crossvaldata_train5, crossvallabels_test5, crossvaldata_test5, reg_param(i));

    crossval_train_error_avg(i) = (crossval_train_error_1 + crossval_train_error_2 + crossval_train_error_3 + crossval_train_error_4 + crossval_train_error_5)/5;
    crossval_test_error_avg(i) = (crossval_test_error_1 + crossval_test_error_2 + crossval_test_error_3 + crossval_test_error_4 + crossval_test_error_5)/5;
end

plot_error_linear_reg_6b_l2(train_error, test_error, crossval_test_error_avg);

%Finding the value of reg_param which gives the least test error.
[minimum_test_error, reg_param_min_index] = min(test_error);
reg_param_min = reg_param(reg_param_min_index);
fprintf('The regularization parameter with minimum testing data error (without cross validation) is %f\n', reg_param_min);

%Finding the value of reg_param which gives the least average
%cross-validation error
[minimum_test_error_cross_val, reg_param_min_index_cross_val] = min(crossval_test_error_avg);
reg_param_min_cross_val = reg_param(reg_param_min_index_cross_val);

fprintf('The regularization parameter with minimum testing data error (with cross validation) is %f\n', reg_param_min_cross_val);

fprintf('Therefore, the chosen value of lambda is %f\n', reg_param_min_cross_val);

%Running the 10% dataset with lambda = 100
[train_error_final, test_error_final, w_final, b_final] = cal_error_linear_reg_6b_l2(trainlabels_10_path, traindata_10_path, testlabels_path, testdata_path, reg_param_min_cross_val);

fprintf('The weight vector is %f %f %f %f %f %f %f %f\n', w_final);

fprintf('The bias term is %f\n', b_final);

fprintf('The training error on the 10percent dataset with lambda = 100 is %f\n', train_error_final);

fprintf('The testing error with the 10percent dataset with lambda = 100 is %f\n', test_error_final);
