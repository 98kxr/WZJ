clear
close all
clc
suma=[];
sumt=[];
summean=[];
Inter = 1;
for INT = 1:Inter

    load('Chen711.mat')%加快读入数据速度，将股票数据保存
    %data = (data-min(data))/(max(data)-min(data));
    %load('kuang_3.mat')
   % load('house_3.mat')
    data=data3;
    DATA_MAX = max(data);
    DATA_MIN = min(data);
    data = (data-min(data))/(max(data)-min(data));
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
    reg = 1e-1;
    a = 0.85;%学习率
    Alpha = 0.25;

    % %结点个数初始化
    % K = 10;%输入结点个数
    % L = 2;%输出结点个数

    %CS
    N = 2^12;%原储备池结点个数
    rat = 1/2;
    R_N = floor(rat*N);%压缩后储备池结点个数
    rng('shuffle')
    Phi= PartHadamardMtx(R_N,N);
    Psi = dwtmtx(N, 'haar', 1);

    x = zeros(N,1);

    %%
    %ESN连接矩阵的初始化
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
    X(R_N, trainLen) = 0;
    Y(outputLen, trainLen) = 0;
    %开始训练
    for i = 1:trainLen
        u = train(1:inputLen,i);
        y = train(inputLen+1:inputLen+outputLen,i);
        x = (1-a)*x + a*tanh(W_in*u + W*x);
    
        %压缩结点
        s = Psi*x;
        CS_x = Phi*s;
     %   CS_x = CS_x./(max(CS_x)-min(CS_x));

        if i > initLen
            U(:,i-initLen) = u;
            X(:,i-initLen) = CS_x;
            Y(:,i-initLen) = y;
        end
    end
    % S = (X*pinv(U))';
    % D = (Y*pinv(U))';
    % W_out = (pinv(S)*D)';
    W_out = Y*X'/(X*X'+reg*eye(R_N));
    coumse=0;
    %%
    %开始测试
    result(trainLen+testLen,1) = 0;
    yy(4,trainLen+testLen) = 0;

    %for j = trainLenLen+1:trainLenLen+testLen
    for j = 1:trainLen+testLen
        u = data(1:inputLen,j);
        y = data(inputLen+1:inputLen+outputLen,j);
        x = (1-a)*x + a*tanh(W_in*u + W*x);

        s = Psi*x;
        CS_x = Phi*s;
     %  CS_x = CS_x./(max(CS_x)-min(CS_x));
        new_y = abs(W_out*CS_x);
      %  new_y=new_y*(DATA_MAX-DATA_MIN)+DATA_MIN;
        result(j) = norm(y-new_y);
        coumse=result(j)*result(j)+coumse;
        yy(1:2, j) = y;
        yy(3:4, j) = new_y;
    end
    allerror=sum(result);
    seterror=sum(result(trainLen:end));
    meanallerror=sum(result)/num;
    suma=[suma allerror];
    sumt=[sumt seterror];
    summean=[summean meanallerror];
    MEAN = sum(result)/num;
    MSE=coumse/num;
    fprintf("总误差值：%f\n测试集误差值：%f\n", sum(result),sum(result(trainLen:end)))
    fprintf("总误差平均值：%f\n测试集误差平均值：%f\n", sum(result)/num,sum(result(trainLen:end))/testLen)
    fprintf("总样本方差：%f\n测试集方差：%f\n", var(result),var(result(trainLen:end)))
   % fprintf("误差超过10的样本个数有：%d\n误差超过10的样本占总样本：%f\n", length(find(result>10)), length(find(result>10))/length(result))
    fprintf("误差超过"+num2str(MEAN)+"的样本个数有：%d\n误差超过"+num2str(MEAN)+"的样本占总样本：%f%%\n", length(find(result>MEAN)), 100*length(find(result>MEAN))/length(result))
end
MEAN = sum(result)/num;
sumresult=sum(result);
sumtestresult=sum(result(trainLen:end));
sumvar= var(result);
sumtestvar=var(result(trainLen:end));
abovemean=length(find(result>MEAN));
aboverate=100*length(find(result>MEAN))/length(result);
%save('1CsES_with_stock_no_noise812.mat','','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate') ;
%save('oldCSESN_with_finance_no_noise919.mat','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate','MEAN'); 
%save('OLDCS_ESN_with_ACL_3_1_new__0220.mat','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate','MEAN'); 
save('OLDCS_ESN_with_Chen3_3_mse_new__0220.mat','','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate','MEAN','MSE','X','result','yy','X','W_out') 
fact_value = yy(1:2,:);
forecast_value = yy(3:4,:);
forecast_value = reshape(forecast_value, [], 1);
fact_value = reshape(fact_value, [], 1);

reshape_data = reshape(data, [], 1);
reshape_forecast_data = data;
reshape_forecast_data(end-1:end, :) = yy(end-1:end, :);
reshape_forecast_data = reshape(reshape_forecast_data, [], 1);

figure
plot(reshape_forecast_data, 'b--', 'DisplayName', 'Forecast data of old CSESN', 'LineWidth', 1.5)
hold on
plot(reshape_data, 'r', 'DisplayName', 'Real data')
xlabel("number of sample")
ylabel("value")
legend
hold off

figure
plot(115400:129500, reshape_forecast_data(115400:129500), 'b--', 'DisplayName', 'Forecast data of old CSESN', 'LineWidth', 1.5)
hold on
plot(115400:129500, reshape_data(115400:129500), 'r', 'DisplayName', 'Real data')
xlabel("number of sample")
ylabel("value")
legend
hold off

figure
plot(792500:801600, reshape_forecast_data(792500:801600), 'b--', 'DisplayName', 'Forecast data of old CSESN', 'LineWidth', 1.5)
hold on
plot(792500:801600, reshape_data(792500:801600), 'r', 'DisplayName', 'Real data')
xlabel("number of sample")
ylabel("value")
legend
hold off

figure
plot([trainLen*2,trainLen*2], [0,1], 'm', 'DisplayName', "The line between training and forecasting")
hold on
plot(forecast_value, 'b--', 'DisplayName', "Forecast data of oldC SESN", 'LineWidth', 1.5)
plot(fact_value, 'r', 'DisplayName', "Real data")
legend
hold off

figure
plot([trainLen,trainLen], [0,0.5], 'm', 'DisplayName', "The line between training and forecasting")
hold on
stem(result(1:trainLen), 'b', 'DisplayName', "Training set error", 'LineWidth', 0.75)
stem(trainLen+1:length(result) ,result(trainLen+1:end), 'r', 'DisplayName', "Test set error")
xlabel("number of sample")
ylabel("error value")
legend
hold off


figure
stem(15060:15110 ,result(15060:15110), 'r', 'DisplayName', "Test set error")
xlabel("number of sample")
ylabel("error value")
legend
hold off
hold off

figure
stem(4315:4353 ,result(4315:4353), 'b', 'DisplayName', "Training set error", 'LineWidth', 0.75)
xlabel("number of sample")
ylabel("error value")
legend
hold off

figure
stem(15060:15110 ,result(15060:15110), 'r', 'DisplayName', "Test set error")
xlabel("number of sample")
ylabel("error value")
legend
hold off