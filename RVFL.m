%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Random Vector Functional Link Networks (Matlab)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2019 ZSD 
%

classdef RVFL
    properties
        Name = 'Random Vector Functional Link Networks';
        version = '1.0 beta';
        L       % hidden node number / start with 1
        W       % input weight matrix
        b       % hidden layer bias vector
        Beta    % output weight vector
        activetype  % the type of activation function
        RegularBeta % the L2 Regularization coefficient of Beta
    end
    methods
        function obj = RVFL(Xtr,Ttr,N_hidden,activetype,RegularBeta)
            
            if ~exist('N_hidden', 'var') || isempty(N_hidden)
                obj.L = 100;
            else
                obj.L = N_hidden;
            end
            if ~exist('activetype', 'var') || isempty(activetype)
                obj.activetype = 'sigmoid';
            else
                obj.activetype = activetype;
            end
            
            if ~exist('RegularBeta', 'var') || isempty(RegularBeta)
                obj.RegularBeta = 0;
            else
                obj.RegularBeta = RegularBeta;
                obj.Name = strcat(obj.Name,' with L2 Regular');
            end
            
            [~, d] = size(Xtr);
            obj.W = rand(d,obj.L)*2 -1;
            obj.b = rand(1,obj.L)*2 -1;
            
            H = obj.GetH(Xtr);
            [obj,~] = obj.ComputeBeta(H, Ttr);
        end
         %% Output Matrix of hidden layer
        function H = GetH(obj, X)
            H = bsxfun(@plus,X*[obj.W],[obj.b]);
            H =  obj.activationFunction(H);
        end
        %% ComputeBeta OLS with or without L2 Regular
        function [obj, Beta] = ComputeBeta(obj, H, T) 
            if obj.RegularBeta ==0
                Beta = pinv(H)*T;
            else
                N_sample = size(T,1);
                if (N_sample >= obj.L)
                    Beta = (H'*H + obj.RegularBeta *eye(obj.L)) \ ( H'*T);
                else
                    Beta = H'*inv(obj.RegularBeta*eye(size(H, 1)) + H * H') *  T; 
                end
            end
            obj.Beta = Beta;
        end
        
        %% Activation Function
         function H = activationFunction(obj, H)
            % Compute activation function
            switch lower(obj.activetype)
                case {'sig','sigmoid'}
                    H = logsig(H);
                case {'tanh'}
                    H = tanh(H);   
                case {'hardlimit'}
                    H = double(hardlim(H));    
                case {'relu'}
                    H(:) = max(H(:),0);
                case {'leakyrelu'}% Leaky ReLUº¯Êý£¨PReLU£©
                    H(H<0) = 0.01*H(H<0);
                    % H(:) = max(H(:),0.01*H(:));
                case {'elu'} %ELU Exponential Linear Units
                    H(H<0) = 0.01*(exp(H(H<0))-1);
                case {'swish'}
                    H = H.*logsig(H);
            end
         end
        %% Get Output
        function O = GetOutput(obj, X)
            H = obj.GetH(X);
            O = H*[obj.Beta];
        end
        %% Get Label
        function O = GetLabel(obj, X)
            O = GetOutput(obj, X);
            O = Tools.OneHotMatrix(O);
        end
        %% Get Accuracy
        function [Rate, O] = GetAccuracy(obj, X, T)
            O = obj.GetLabel(X);
            Rate = 1- confusion(T',O');
        end
        %% Get Error, Output and Hidden Matrix
        function [Error, O, H, E] = GetResult(obj, X, T)
            % X, T are test data or validation data
            H = obj.GetH(X);
            O = H*(obj.Beta);
            E = T - O;
            Error =  Tools.RMSE(E);
        end
    end
end