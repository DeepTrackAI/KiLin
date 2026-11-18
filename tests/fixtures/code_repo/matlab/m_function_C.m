function result = m_function_C(a, b)
% Return the larger of py_function_A(a, b) and m_function_A([a b]),
% calling the Python calculator from MATLAB.
from_python = py.calculator_py.py_function_A(a, b);
result = max(double(from_python), m_function_A([a b]));
end
