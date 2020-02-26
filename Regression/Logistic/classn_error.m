function err = classn_error(predy, y)
% To compute the classification error. 
	err = 1-length(find(predy==y))/length(y);
end