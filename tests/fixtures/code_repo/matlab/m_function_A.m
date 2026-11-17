function result = m_function_A(values)
% Return the mean of VALUES, using m_function_B for the sum.
total = m_function_B(values);
result = total / numel(values);
end
