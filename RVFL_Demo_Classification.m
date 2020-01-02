% Demo Classification of SCN
clear;
clc;
close all; 
format long;
load('Demo_Iris.mat');
 
%% show some samples
% X: each row vector represents one sample input.
% T: each row vector represents one sample label.
% same to the X2 and T2.
n1 = randperm(120, 5); % Randomly select 5 samples from training set.
disp('Training Input:')
disp(X(n1,:))
disp('Training Label:')
disp(T(n1,:))

n2 = randperm(30, 2);  % Randomly select 2 samples from test set.
disp('Test Input:')
disp(X2(n2,:))
disp('Test Label:')
disp(T2(n2,:))

%% Parameter Setting
N_hidden = 100;
activetype ='tanh'; % °üÀ¨sig tanh hardlimit
%% Model Training
M = RVFL(X,T,N_hidden,activetype);
disp(M);
%% Model output vs target on training dataset
O1 = M.GetLabel(X);
disp(['Training Acc: ' num2str(M.GetAccuracy(X, T))]);
figure;
plotconfusion(T',O1','Training  ');

%% Model output vs target on test dataset
O2 = M.GetLabel(X2);
disp(['Test Acc: ', num2str(M.GetAccuracy(X2, T2))]);
figure;
plotconfusion(T2',O2','Test  ');

% The End










