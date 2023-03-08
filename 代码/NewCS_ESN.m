clear
close all
clc
Inter = 1;
suma=[];
sumt=[];
summean=[];

for INT = 1:Inter

load('finance_3.mat')%加快读入数据速度，将股票数据保存
%data=data3;
%data = data./(max(data)-min(data));
%load('kuang_3.mat')
data = (data-min(data))/(max(data)-min(data));
DATA_MAX = max(data);
DATA_MIN = min(data);
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
cross=1;
step=2;
    %CS
    N = 2^12;%原储备池结点个数
    rat = 1/2;
    R_N = floor(rat*N);%压缩后储备池结点个数
    rng('shuffle')
%Phi= PartHadamardMtx(floor(R_N/cross),floor(N/cross));%哈米达
Phi=sqrt(1/R_N)*randn(R_N,N);
  %  Psi = dwtmtx(floor(N/cross), 'haar', 1);
  % Psi=fft(eye(N,N));

    x =zeros(N,1);

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
%pP=pinv(Psi);
%pP=Psi;
    %%
    U(inputLen,trainLen) = 0;
    X(R_N, trainLen) = 0;
    fprintf("%d,%d\n",size(X))
    Y(outputLen, trainLen) = 0;
    %开始训练
   % pP=pinv(Phi*Psi);
    for i = 1:trainLen
        u = train(1:inputLen,i);
       % u = u./(DATA_MAX-DATA_MIN);
       % u = (u-DATA_MIN)/(DATA_MAX-DATA_MIN);
      
        y = train(inputLen+1:inputLen+outputLen,i);
       % y = y./(DATA_MAX-DATA_MIN);
       % y = (y-DATA_MIN)/(DATA_MAX-DATA_MIN);
        x = (1-a)*x + a*tanh(W_in*u + W*x);
       Sp=fft(x);
       %Sp=Psi*x;
        CS_x = Phi*Sp;
    fprintf("CS_X %d,%d\n",size(CS_x))
        if i > initLen
            U(:,i-initLen) = u;
            X(:,i-initLen) = CS_x;
            Y(:,i-initLen) = y;
        end
    end
    
   
  %  CS_X = CS_X./(max(CS_X)-min(CS_X));
W_out = Y*X'/(X*X'+reg*eye(R_N));
fprintf("stop right here %d %d",size(W_out));
fprintf(" %d %d",size(x));
    %%
    %开始测试
    result(trainLen+testLen,1) = 0;
    yy(4,trainLen+testLen) = 0;
    coumse=0;
    %for j = trainLenLen+1:trainLenLen+testLen
    for j = 1:trainLen+testLen
        u = data(1:inputLen,j);
        %u = u./(DATA_MAX-DATA_MIN);
        % u = (u-DATA_MIN)/(DATA_MAX-DATA_MIN);
        y = data(inputLen+1:inputLen+outputLen,j);
       % y = y./(DATA_MAX-DATA_MIN);
      % y = (y-DATA_MIN)/(DATA_MAX-DATA_MIN);
        x = (1-a)*x + a*tanh(W_in*u + W*x);
       S=fft(x);
        %S=pP*x;
        CS_x=Phi*S;
      %  CS_x = CS_x./(max(CS_x)-min(CS_x));
         new_y = abs(W_out*CS_x);
fprintf(" %d %d\n",size(x));
    %if new_y > DATA_MAX
     %   new_y = DATA_MAX;
    %end
    %if new_y < DATA_MIN
     %   new_y = DATA_MIN;
    %end
        result(j) = norm(y-new_y);
        coumse=result(j)*result(j)+coumse;
        yy(1:2, j) = y;
        yy(3:4, j) = new_y;
    end
    MEAN = sum(result)/num;
    MSE=coumse/num;
    fprintf("总误差值：%f\n测试集误差值：%f\n", sum(result),sum(result(trainLen:end)))
    fprintf("总误差平均值：%f\n测试集误差平均值：%f\n", sum(result)/num,sum(result(trainLen:end))/testLen)
    fprintf("总样本方差：%f\n测试集方差：%f\n", var(result),var(result(trainLen:end)))
    fprintf("误差超过"+num2str(MEAN)+"的样本个数有：%d\n误差超过"+num2str(MEAN)+"的样本占总样本：%f%%\n", length(find(result>MEAN)), 100*length(find(result>MEAN))/length(result))
%     fprintf("误差超过10的样本个数有：%d\n误差超过10的样本占总样本：%f\n", length(find(result>10)), length(find(result>10))/length(result))
end
sumresult=sum(result);
sumtestresult=sum(result(trainLen:end));
sumvar= var(result);
sumtestvar=var(result(trainLen:end));
abovemean=length(find(result>MEAN));
aboverate=100*length(find(result>MEAN))/length(result);
%save('ESN_with_nonoise_stock812.mat','','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate') 
%figure
%stem(trainLen+5e2:trainLen+6e2 ,result(trainLen+5e2:trainLen+6e2), 'b', 'DisplayName', "test set error by ESN without white noise")
%xlabel("number of sample")
%ylabel("error value")
%legend('Location', 'northwest')
%hold off
MEAN = sum(result)/num;
sumresult=sum(result);
sumtestresult=sum(result(trainLen:end));
sumvar= var(result);
sumtestvar=var(result(trainLen:end));
abovemean=length(find(result>MEAN));
aboverate=100*length(find(result>MEAN))/length(result);
%save('1CsES_with_Chen3_919.mat','','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate','MEAN') ;

save('NewCsES_with_finance_3_01_maxmin_u_y_0220.mat','','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate','MEAN','MSE','X','result','yy','X','W_out') 

%save('New1CsES_with_kuang_3_1205.mat','','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate','MEAN') ;

fact_value = yy(1:2,:);
forecast_value = yy(3:4,:);
forecast_value = reshape(forecast_value, [], 1);
fact_value = reshape(fact_value, [], 1);

reshape_data = reshape(data, [], 1);
reshape_forecast_data = data;
reshape_forecast_data(end-1:end, :) = yy(end-1:end, :);
reshape_forecast_data = reshape(reshape_forecast_data, [], 1);

figure
plot(reshape_forecast_data, 'b--', 'DisplayName', 'Forecast data of New-CSESN', 'LineWidth', 1.5)
hold on
plot(reshape_data, 'r', 'DisplayName', 'Real data')
xlabel("number of sample")
ylabel("value")
legend
hold off

figure
plot(115400:129500, reshape_forecast_data(115400:129500), 'b--', 'DisplayName', 'Forecast data of New-ESN', 'LineWidth', 1.5)
hold on
plot(115400:129500, reshape_data(115400:129500), 'r', 'DisplayName', 'Real data')
xlabel("number of sample")
ylabel("value")
legend
hold off

figure
plot(792500:801600, reshape_forecast_data(792500:801600), 'b--', 'DisplayName', 'Forecast data of New-ESN', 'LineWidth', 1.5)
hold on
plot(792500:801600, reshape_data(792500:801600), 'r', 'DisplayName', 'Real data')
xlabel("number of sample")
ylabel("value")
legend
hold off

figure
plot([trainLen*2,trainLen*2], [0,1], 'm', 'DisplayName', "The line between training and forecasting")
hold on
plot(forecast_value, 'b--', 'DisplayName', "Forecast data of New-ESN", 'LineWidth', 1.5)
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