% This is the main code, which has the path for the data and labels.
% The plot function is called here.

trainlabels_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-4\Spambase\y_train.txt';
traindata_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-4\Spambase\X_train.txt';
testlabel_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-4\Spambase\y_test.txt';
testdata_path = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-4\Spambase\X_test.txt';

%Importing Cross Validation data
cross_val_path_root = 'C:\Penn\Courses\Semester2\CIS 520\Homework\HW1\ps1_kit\Problem-4\Spambase\Cross-validation\Fold';
cross_val_path_data = '\X.txt';
cross_val_path_labels = '\y.txt';

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

crossvallabels_train = zeros(5, 200, 1);
crossvallabels_test = zeros(5, 50, 1);
crossvaldata_train = zeros(5, 200, 57);
crossvaldata_test = zeros(5, 50, 57);

crossvallabels_train(1,:,:) = [crossvallabels1; crossvallabels2; crossvallabels3; crossvallabels4];
crossvallabels_train(2,:,:) = [crossvallabels2; crossvallabels3; crossvallabels4; crossvallabels5];
crossvallabels_train(3,:,:) = [crossvallabels3; crossvallabels4; crossvallabels5; crossvallabels1];
crossvallabels_train(4,:,:) = [crossvallabels4; crossvallabels5; crossvallabels1; crossvallabels2];
crossvallabels_train(5,:,:) = [crossvallabels5; crossvallabels1; crossvallabels2; crossvallabels3];
crossvaldata_train(1,:,:) = [crossvaldata1; crossvaldata2; crossvaldata3; crossvaldata4]; 
crossvaldata_train(2,:,:) = [crossvaldata2; crossvaldata3; crossvaldata4; crossvaldata5]; 
crossvaldata_train(3,:,:) = [crossvaldata3; crossvaldata4; crossvaldata5; crossvaldata1]; 
crossvaldata_train(4,:,:) = [crossvaldata4; crossvaldata5; crossvaldata1; crossvaldata2]; 
crossvaldata_train(5,:,:) = [crossvaldata5; crossvaldata1; crossvaldata2; crossvaldata3];
crossvallabels_test(1,:,:) = crossvallabels5;
crossvallabels_test(2,:,:) = crossvallabels1;
crossvallabels_test(3,:,:) = crossvallabels2;
crossvallabels_test(4,:,:) = crossvallabels3;
crossvallabels_test(5,:,:) = crossvallabels4;
crossvaldata_test(1,:,:) = crossvaldata5;
crossvaldata_test(2,:,:) = crossvaldata1;
crossvaldata_test(3,:,:) = crossvaldata2;
crossvaldata_test(4,:,:) = crossvaldata3;
crossvaldata_test(5,:,:) = crossvaldata4;



train_error = ones(8, 1);
test_error = ones(8, 1);

reg_param = [10^-7 10^-6 10^-5 10^-4 10^-3 10^-2 10^-1 1];

for i = 1:8
    [train_error(i), test_error(i)] = cal_error_l2_reg(trainlabels_path, traindata_path, testlabel_path, testdata_path, reg_param(i));
end

% 5-fold cross validation
crossval_train_error = ones(5, 1);
crossval_test_error = ones(5, 1);
crossval_train_error_avg = ones(8, 1);
crossval_test_error_avg = ones(8, 1);
for i = 1:8
    for j = 1:5 
        [crossval_train_error(j), crossval_test_error(j)] = cal_error_l2_reg_crossval(crossvallabels_train(j,:,:), crossvaldata_train(j,:,:), crossvallabels_test(j,:,:), crossvaldata_test(j,:,:), reg_param(i));
    end
    crossval_train_error_avg(i) = mean(crossval_train_error);
    crossval_test_error_avg(i) = mean(crossval_test_error);
end

plot_error_l2_reg(train_error, test_error, crossval_test_error_avg);


for i = 1:8
    fprintf('The average cross-validation error for each value of lambda is %f\n', crossval_test_error_avg(i));
end

%Finding the value of reg_param which gives the least test error.
[minimum_test_error, reg_param_min_index] = min(test_error);
reg_param_min = reg_param(reg_param_min_index);
fprintf('The regularization parameter with minimum testing data error (without cross validation) is %f\n', reg_param_min);

%Finding the value of reg_param which gives the least average
%cross-validation error
[minimum_test_error_cross_val, reg_param_min_index_cross_val] = min(crossval_test_error_avg);
reg_param_min_cross_val = reg_param(reg_param_min_index_cross_val);

fprintf('The regularization parameter with minimum testing data error (with cross validation) is %f\n', reg_param_min_cross_val);

fprintf('The corresponding training error for lambda 0.000001 is %f\n', train_error(2));
fprintf('The corresponding test error for lambda 0.000001 is %f\n', test_error(2));