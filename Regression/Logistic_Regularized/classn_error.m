function error = classn_error(predicted_y, y)
% To compute the classification error. 
	error = 1 - length(find(predicted_y == y))/length(y);
end