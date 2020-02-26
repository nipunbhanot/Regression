function [res] = mean_squared_error_6b(y, y_hat)
  res = sum((y - y_hat).^2) / length(y);
end
