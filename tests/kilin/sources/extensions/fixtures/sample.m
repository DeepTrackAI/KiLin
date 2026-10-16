% Synthetic MATLAB class for extraction tests.
classdef (Abstract) Sample < Base
    properties
        Value
    end
    methods (Static)
        function obj = Sample(value)
            obj.Value = value;
        end
        function [first, second] = compute(obj)
            first = obj.Value;
            second = first + 1;
        end
    end
end
