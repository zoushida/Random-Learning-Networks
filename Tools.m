classdef Tools
    properties
    end
    methods (Static = true)
        % Transfer the label {1,2,3,4,...} to [0,...,1,...,0] ...
        function X = LabelToMatrix(x, k)
            % transfer the label {1,2,3,4,...} to [0,...,1,...,0] ...
            N = length(x);
            X = zeros(N,k);
            for i = 1:N
                X(i,x(i)) = 1;
            end
        end
        %% 获取kfold 当前的训练和测试集
        function [X_train,T_train,X_test,T_test] = Shuffle_kflod(X,T,seed,K,k_i)
            [n,~] = size(X);
            testIndex = zeros(n,1);
            rng(seed);
            newIndex = randperm(n);
            newX = X(newIndex,:);
            newT = T(newIndex,:);
            if k_i<K
                testIndex(floor(n*(k_i-1)/K)+1:floor(n*(k_i)/K)) =1;
            elseif k_i==K
                testIndex(floor(n/K)*(k_i-1):end) =1;
            end
            testIndex = (testIndex ==1);
            X_train = newX(~testIndex,:);
            T_train = newT(~testIndex,:);
            X_test = newX(testIndex,:);
            T_test = newT(testIndex,:);
        end
        
        %% 打乱样本顺序并生成训练和验证集合
        function [X_train,T_train,X_val,T_val] = ShuffleData(X,T,rate,seed)
            if exist('seed','var') || ~isempty(seed)
                rng(seed);
            end
            [n,~] = size(X);
            num_train = floor(rate*n);
            
            newIndex = randperm(n);
            newX = X(newIndex,:);
            newT = T(newIndex,:);
            X_train = newX(1:num_train,:);
            T_train = newT(1:num_train,:);

            X_val = newX(num_train+1:end,:);
            T_val = newT(num_train+1:end,:);
        end
        %% 归一化数据 mapminmax
        function [X_train,X_test]= isMapminmaxNormal(X_train,X_test)
            [X_train,PS] = mapminmax(X_train');
            X_test = mapminmax('apply',X_test',PS);
            X_train = X_train';
            X_test  = X_test';
        end

        %% 归一化数据 Autoscales
        function [X_train,X_test]= isautoNormal(X_train,X_test)
            [X_train,xmeans,stdx] = Tools.auto01(X_train);
            X_test = Tools.scale_001(X_test,xmeans,stdx);
        end

        function [ax,mx,stdx] = auto01(x)
            %AUTO Autoscales matrix to mean zero unit variance
            %  Autoscales a matrix (x) and returns the resulting matrix (ax)
            %  with mean-zero unit variance columns, a vector of means (mx)
            %  and a vector of standard deviations (stdx) used in the scaling.
            %
            %I/O:  [ax,mx,stdx] = auto(x);
            %
            %See also: MDAUTO, MDMNCN, MDRESCAL, MDSCALE, MNCN, SCALE, RESCALE

            %Copyright Eigenvector Research, Inc. 1991-98
            %Modified 11/93
            %Checked on MATLAB 5 by BMW  1/4/97

            [m,n] = size(x);
            mx    = mean(x);
            stdx  = std(x);
            for i=1:n
                if stdx(1,i)==0
                    ax(:,i)=x(:,i)-mx(ones(m,1),i);
                else
                    ax(:,i)    = (x(:,i)-mx(ones(m,1),i))./stdx(ones(m,1),i);
                end
            end         
        end
        function sx = scale_001(x,means,stds)
            %SCALE Scales matrix as specified.
            %  Scales a matrix (x) using means (mx) and standard
            %  deviations (stds) specified.
            %
            %I/O format is:  sx = scale(x,mx,stdx);
            %
            %  If only two input arguments are supplied then the function
            %  will not do variance scaling, but only vector subtraction.
            %
            %I/O format is:  sx = scale(x,mx);
            %
            %See also: AUTO, MDAUTO, MDMNCN, MDRESCAL, MDSCALE, MNCN, RESCALE

            %Copyright Eigenvector Research, Inc. 1991-98
            %Modified 11/93
            %Checked on MATLAB 5 by BMW  1/4/97
            [m,n] = size(x);
            if nargin == 3
                for i=1:n
                    if stds(1,i)~=0
                        sx(:,i) = (x(:,i)-means(ones(m,1),i))./stds(ones(m,1),i);
                    else
                        sx(:,i) = (x(:,i)-means(ones(m,1),i));
                    end
                end
            end
        end
        
        
        %% 回归曲线
        function RegressionCurve(Target,Fitting,titleName)
        %     figure
            plot(Target)
            hold on;
            plot(Fitting)
            xlabel('样本');
            ylabel('函数值');
            legend('Target', 'Model Output');
            if exist('titleName', 'var') 
                title(titleName);
            end
        end
        %% 显示RMSE
        function value = ViewRMSE(Target,Fitting,str)
            value = Tools.RMSE(Target-Fitting);
            disp(strcat(str,num2str(value)));
        end
        %% CalculateError
        function Error = CalculateError(E0)
            % get the RMSE
            [EN, ~]  = size(E0);
            Error = sqrt(sum(sum( E0.^2)/EN)); % multi-output
        end
        
        %% RMSE
        function Error = RMSE(E0)
            % get the RMSE
            [EN, ~]  = size(E0);
            Error = sqrt(sum(sum( E0.^2)/EN)); % multi-output
        end        
 
 
        %Split the data into two parts
        function [PartA, PartB] = SplitData(Data,Rate,seed)
            % each row is a sampleS
            N = size(Data,1);% get the sample number
            switch nargin
                case 3  % shuffle the data
                    rng(seed);
                    splitpoint = floor(N*Rate);% find the split position
                    Data = Data(randperm(N),:);
                    PartA = Data(1:splitpoint,:);
                    PartB = Data(splitpoint+1:end,:);
                case 2  % No shuffle the data
                    splitpoint = floor(N*Rate);% find the split position
                    PartA = Data(1:splitpoint,:);
                    PartB = Data(splitpoint+1:end,:);
                case 1
                    Rate = 0.5;
                    splitpoint = floor(N*Rate);% find the split position
                    PartA = Data(1:splitpoint,:);
                    PartB = Data(splitpoint+1:end,:);
                otherwise
                    disp('Wrong arguments');
            end
            if ~exist('DispON', 'var') || isempty(DispON)
                DispON = true;
            else
                DispON = false;
            end            
            %disp(['SplitRate  : ', num2str(Rate)]);
            %disp(['Part A size: ', num2str(splitpoint),  ' ',num2str(splitpoint/N*100),'% ', Tools.ByteSize(PartA)]);
            %disp(['Part B size: ', num2str(N-splitpoint),' ' ,num2str((N-splitpoint)/N*100),'% ',Tools.ByteSize(PartB)]);
        end
        
        %% Normalize a data matrix
        % Note: for this tools, assuming each row is a sample       
        function [Xn, PS] = Norm(X, Ymin, Ymax)
            %disp('*Tip: X should be a N-by-P matrix.');
            %disp('N is sample number, P is feature size!');
            X = X';
            [Xn,PS] = mapminmax(X,Ymin,Ymax); % using
            Xn = Xn';
        end
        
        % using the same setting of evaluation data or test data
        function [Xn] = NormApply(X, PS)
            X = X';
            Xn = mapminmax('apply',X, PS);
            Xn = Xn';
        end
        
        % To convert these outputs back into the same units
        % that were used for the original targets use the settings
        function [X] = NormReverse(Xn, PS)
            Xn = Xn';
            X = mapminmax('reverse',Xn, PS); % using
            X = X';
        end
        
        % Standardized z-scores
        % columns of X are centered to have mean 0 and scaled to have standard deviation 1.
        % Z is the same size as X.
        function [Z,mu,sigma] = ZscoreData(X)            
            [Z,mu,sigma]= zscore(X);
            % user could overload this function
            % also returns the means and standard deviations used for centering and scaling,
        end
        
        % BYTESIZE writes the memory usage of the provide variable to the given file
        % identifier. Output is written to screen if fid is 1, empty or not provided.
        function str = ByteSize(in, fid)
            if nargin == 1 || isempty(fid)
                fid = 1;
            end            
            s = whos('in');
            %fprintf(fid,[Bytes2str(s.bytes) '\n']);
            str =  Tools.Bytes2str(s.bytes);
        end    
        
        % BYTES2STR Private function to take integer bytes and convert it to
        % scale-appropriate size.          
        function str = Bytes2str(NumBytes)
          
            scale = floor(log(NumBytes)/log(1024));
            switch scale
                case 0
                    str = [sprintf('%.0f',NumBytes) ' b'];
                case 1
                    str = [sprintf('%.2f',NumBytes/(1024)) ' kb'];
                case 2
                    str = [sprintf('%.2f',NumBytes/(1024^2)) ' Mb'];
                case 3
                    str = [sprintf('%.2f',NumBytes/(1024^3)) ' Gb'];
                case 4
                    str = [sprintf('%.2f',NumBytes/(1024^4)) ' Tb'];
                case -inf
                    % Size occasionally returned as zero (eg some Java objects).
                    str = 'Not Available';
                otherwise
                    str = 'Over a petabyte!!!';
            end
        end
        
        %% Transfer each row of a matrix into a one-hot vector
        % The maximum values is 1, everything else is zero.
        function Y = OneHotMatrix(X)
            [N,p] = size(X);
            Y = zeros(N,p);
            if p>1
                for i = 1:N
                    [~,ind] = max(X(i,:));
                    Y(i,ind) = 1;
                end
            else
                for i = 1:N
                    if X(i) > 0.50
                        Y(i) = 1;
                    end
                end
            end
        end 
        
        %% Create GIF
        function res = WriteGIF(name,frames,dt)
            nframe = length(frames);            
            for i=1:nframe
                [image,map] = frame2im(frames(i));
                [im,map2]   =  rgb2ind(image,256);
                if i==1
                    imwrite(im,map2,name,'GIF','WriteMode','overwrite','DelayTime',dt,'LoopCount',inf);
                else
                    imwrite(im,map2,name,'WriteMode','append','DelayTime',dt); %,'LoopCount',inf);
                end
            end
            res = true;
        end
        
    end
end