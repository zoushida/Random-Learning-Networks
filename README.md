# Random Vector Functional Link Networks (Matlab)
可用于分类或回归问题

N_hidden = 100;
activetype ='tanh'; % 包括sig tanh hardlimit等
M = RVFL(X,T,N_hidden,activetype); % modelling and training
% 01.Model output vs target on test dataset for Classification
O1 = M.GetLabel(X);
figure;
plotconfusion(T',O1','Training  ');

% 02.Model output vs target on test dataset for Regression
O2 = M.GetOutput(X2);
figure;
plot(X2, T2, 'r.-'); hold on;
plot(X2, O2, 'b.-');  
xlabel('X2');
ylabel('T2');
legend('Test Target', 'Model Output');
