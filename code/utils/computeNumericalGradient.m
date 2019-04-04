function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

epsilon = 1e-4;
for i = 1:size(theta)%100
    e = zeros(size(theta));
    e(i) = epsilon;
    theta_pos = theta + e;
    theta_neg = theta - e;
    numgrad(i) = (J(theta_pos)-J(theta_neg))/2/epsilon;
end

end
