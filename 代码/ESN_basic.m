clear
close all
clc
load('finance_3.mat')%加快读入数据速度，将股票数据保存。该数据进行了标准化处理
%data=data(1:10000);
%data=data3;
%data=coll;
fprintf("%d \n",size(data));
DATA_MAX = max(data);
DATA_MIN = min(data);
%data = (data-min(data))/(max(data)-min(data));
%%
%训练条件的初始化
inputLen = 50;%输入样本长度
outputLen = 1;%输出样本长度
Len = inputLen+outputLen;%每个训练样本长度

num = floor(length(data)/Len); %所有样本个数
mo = mod(length(data), Len);%多余数据个数

trainLen = ceil(num*9/10);%训练样本个数
testLen = num-trainLen; %测试样本个数
initLen = 5;%初始化样本个数

fprintf("\t所有样本个数：%d\n\t训练集个数：%d\n\t初始化样本个数：%d\n\t测试集样本个数：%d\n", num, trainLen, initLen, testLen);

data = data(1:end-mo);%去除多余的数据
data = reshape(data, Len, []);
fprintf("data 大小%d\n", size(data));
% %修改测试样本
% Start = 1;%从第Start个样本一直到Start+testLen-1样本都是新的测试样本（总共testLen个样本）。
% Temp = data(:,Start:Start+testLen-1);
% data(:,Start:Start+testLen-1) = data(:, end-testLen+1:end);
% data(:, end-testLen+1:end) = Temp;
% clear Temp

train = data(:,1:trainLen);%训练集
forecast = data(:,trainLen+1:num);%测试集
%以上代码给出了测试集和测试。下面我们开始构建CS-ESN神经网络。

%%
%ESN神经网络
%reg = 0;
%a = 0.3;%学习率
%Alpha = 0.95;
    reg = 1e-1;
    a = 0.85;%学习率
    Alpha = 0.25;
% %结点个数初始化
% K = 10;%输入结点个数
%N = 2^10;%储备池结点个数_stock
% L = 2;%输出结点个数
N = 2^12;
%连接矩阵的初始化
x = ones(N,1);
W_in = rand(N,inputLen) - 0.5;
W = sprand(N,N,0.01);
W = W + W';
W_mask = (W~=0);
W(W_mask) = (W(W_mask)-0.5);
rhoW = abs(eigs(W,1));
W = Alpha/rhoW .* W;
%W_out = rand(L,N) - 0.5;
%激活函数为：tanh（tanh）

%%
U(inputLen,trainLen) = 0;
X(N, trainLen) = 0;
Y(outputLen, trainLen) = 0;
%开始训练
for i = 1:trainLen
    u = train(1:inputLen,i);
    y = train(inputLen+1:inputLen+outputLen,i);
    x = (1-a)*x + a*tanh(W_in*u + W*x);
  %  x = x+0.3;
   % x = 2.5.*x; 
%     x = round(1.499.*x);
    if i > initLen
        U(:,i-initLen) = u;
        X(:,i-initLen) = x;
        Y(:,i-initLen) = y;
    end
end
% S = (X*pinv(U))';
% D = (Y*pinv(U))';
% W_out = (pinv(S)*D)';
W_out = Y*X'/(X*X'+reg*eye(N));

%%
%开始测试
result(trainLen+testLen,1) = 0;
yy(4,trainLen+testLen) = 0;
coumse=0;
%for j = trainLenLen+1:trainLenLen+testLen
for j = 1:trainLen+testLen
    u = data(1:inputLen,j);
    y = data(inputLen+1:inputLen+outputLen,j);
    x = (1-a)*x + a*tanh(W_in*u + W*x);
   % x = x + 0.3;
   % x = 2.5.*x; 
%     x = round(1.499.*x);
    new_y = abs(W_out*x);
    %激活函数
    result(j) = norm(y-new_y);
    coumse=result(j)*result(j)+coumse;
   % result(j) = abs(y-new_y);
   % fprintf("%d\n", size(new_y));
    yy(1:2, j) = y;
    yy(3:4, j) = new_y;
end
       MEAN = sum(result)/num;
    MSE=coumse/num;
%%
sumresult=sum(result);
sumtestresult=sum(result(trainLen:end));
sumvar= var(result);
sumtestvar=var(result(trainLen:end));
abovemean=length(find(result>MEAN));
aboverate=100*length(find(result>MEAN))/length(result);
fprintf("总误差值：%f\n测试集误差值：%f\n", sum(result),sum(result(trainLen:end)))
fprintf("总误差平均值：%f\n测试集误差平均值：%f\n", sum(result)/num,sum(result(trainLen:end))/testLen)
fprintf("总样本方差：%f\n测试集方差：%f\n", var(result),var(result(trainLen:end)))
% fprintf("误差超过10的样本个数有：%d\n误差超过10的样本占总样本：%f\n", length(find(result>10)), length(find(result>10))/length(result))
%save('ESN_with_allstock_new_no01_1224.mat','','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate','MEAN','MSE','result','yy','X','W_out') 
%%
%可视化展示

