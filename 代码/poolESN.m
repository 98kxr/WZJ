%SN自编较好代码

%训练代码
clear
close all
clc
suma=[];
sumt=[];
summean=[];
% % %%
% % %读入数据
% % first = "E:\LOVE\大创项目\股票数据\上证A股\SH (";
% % last = ").csv";
% % filename = 2;
% % warning off
% % file = first + num2str(filename)+last;
% % temp = readtable(file, "ReadVariableNames", true, "Delimiter", ",");
% % data = table2array(temp(:,8));
% % warning on
% % clear temp first last file
% % %以上代码将第一只股票数据保存到data数据中。下面我们每inputLen个数据为一组，进行训练和预测。outputLen个数据输入，两个数据输出。
% 
% %%
% %新型读入数据，读入多个文件数据
% first = "E:\LOVE\大创项目\股票数据\上证A股\SH (";
% last = ").csv";
% filename_start = 1;
% filename_end = 200;
% warning off
% 
% data = [];
% for filename = filename_start:filename_end
%     file = first + num2str(filename)+last;
%     temp = readtable(file, "ReadVariableNames", true, "Delimiter", ",");
%     data_temp = table2array(temp(:,8));
%     data_temp = data_temp(data_temp<60);
%     data = cat(1, data, data_temp);
% end
% warning on
% clear temp first last file
% 
% % %%
% % %生成数据
% % %Logistic混沌系统，让系统先迭代一定次数后，再使用生成的值。
% % x = 0.2;%初始值
% % t0 = 1900;%预先迭代次数
% % t = 50000;%所需要的数据量
% % u = 3.9999;%Logistic参数
% % 
% % Logistic = zeros(1,t);
% % for s0 = 1:t0
% %     x = u*x*(1-x);
% % end
% % for s = 1:t
% %     x = u*x*(1-x);
% %     Logistic(s) = x;
% % end
% % data = Logistic';
% % clear Logistic
% 
%load('allstock.mat')%加快读入数据速度，将股票数据保存。该数据进行了标准化处理
load('finance_3.mat')
%data=data1;
%data = data./(max(data)-min(data));
%data = (data-min(data))/(max(data)-min(data));
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
N = 2^12;%储备池结点个数
% L = 2;%输出结点个数

%连接矩阵的初始化
x = zeros(N,1);
W_in = rand(N,inputLen) - 0.5;
W = sprand(N,N,0.01);
W = W + W';
W_mask = (W~=0);
W(W_mask) = (W(W_mask)-0.5);
rhoW = abs(eigs(W,1));
W = Alpha/rhoW .* W;
%W_out = rand(L,N) - 0.5;
%激活函数为：tanh（tanh）

%xstep=2;
%ystep=2;
cross=1;
step=2;
%%
U(inputLen,trainLen) = 0;
X(floor(N/cross), trainLen) = 0;
Y(outputLen,trainLen) = 0;
%开始训练
for i = 1:trainLen
    u = train(1:inputLen,i);
    y = train(inputLen+1:inputLen+outputLen,i);
    x = (1-a)*x + a*tanh(W_in*u + W*x);
  x = x + 0.55;%poolcsesn_stock
  % x = 0.3.*x;%poolcsesn_stock
  x = 1.5.*x;
    fx=x;
        xp=[];
        for ii = 1:cross:N-cross+1
        xc=[];
        for ic=ii:ii+step-1
        %if ii+cross>N-cross+1
        %for coui=1:ii+cross-N-cross+1
        %fx(end+1)=0;
        %end
        %end
        if ic>N-cross+1
        for couj=ic-N-cross+1
        fx(end+1)=0;
        end
        end
        xc=[xc fx(ic)];
        
        %xc=mean(xc);
        end
        xa=mean(xc);

        %xa=mean(xc);
        
        xp=[xp xa];
        end
        xp=reshape(xp,length(xp),[]);
        xp=xp*8.5;
        

    if i > initLen
        U(:,i-initLen) = u;
        X(:,i-initLen) = xp;
        Y(:,i-initLen) = y;
    end
end


%     x = round(1.499.*x);
%mo_Xx=mod(length(X(1,:)), xstep);
%mo_Xy=mod(length(X(:,1)), ystep);
%Xnew=zeros(N+mo_Xy,trainLen+mo_Xx);
%for xc=1:N
 %   for xr=1:trainLen
  %     Xnew(xc,xr)=X(xc,xr); 
   % end
%end
%X=Xnew;
%X=X(1:N-mo_Xy,1:trainLen-mo_Xx);

%mo_Yx=mod(length(Y(1,:)),xstep);
%mo_Yy=mod(length(Y(:,1)),ystep);
%Y=Y(1:outputLen-mo_Yy,1:trainLen-mo_Yx);
%Y=Y(:,1:trainLen-mo_Yx);
%池化

% S = (X*pinv(U))';
% D = (Y*pinv(U))';
% W_out = (pinv(S)*D)';
W_out = Y*X'/(X*X'+reg*eye(floor(N/cross)));
 %fprintf("%d\n", size(W_out));
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
     x = x + 0.55;
      %  x = 0.3.*x;
      x = 1.5.*x;
    fx=x;
        xp=[];
        for ii = 1:cross:N-cross+1
        xc=[];
        for ic=ii:ii+step-1
        %if ii+cross>N-cross+1
        %for coui=1:ii+cross-N-cross+1
        %fx(end+1)=0;
        %end
        %end
        if ic>N-cross+1
        for couj=ic-N-cross+1
        fx(end+1)=0;
        end
        end
        xc=[xc fx(ic)];
   
        %xc=mean(xc);
        end
       % xa=mean(xc);
        xa=mean(xc);
        xp=[xp xa];
        end
       
    xp=reshape(xp,length(xp),[]);
    xp=xp*8.5;
    %fprintf("%d\n", size(xp));
    new_y = abs(W_out*xp);
    %fprintf("%d\n", size(new_y));
    %激活函数
    % if new_y > DATA_MAX
       % new_y = DATA_MAX;
    %end
    %if new_y < DATA_MIN
     %   new_y = DATA_MIN;
   %                  end
       % result(j) = norm(y-new_y);
       result(j) = norm(y-new_y);
        coumse=result(j)*result(j)+coumse;
        yy(1:2, j) = y;
        yy(3:4, j) = new_y;
    end

    MEAN = sum(result)/num;
    MSE=coumse/num;
%%
fprintf("总误差值：%f\n测试集误差值：%f\n", sum(result),sum(result(trainLen:end)))
fprintf("总误差平均值：%f\n测试集误差平均值：%f\n", sum(result)/num,sum(result(trainLen:end))/testLen)
fprintf("总样本方差：%f\n测试集方差：%f\n", var(result),var(result(trainLen:end)))
 fprintf("误差超过"+num2str(MEAN)+"的样本个数有：%d\n误差超过"+num2str(MEAN)+"的样本占总样本：%f%%\n", length(find(result>MEAN)), 100*length(find(result>MEAN))/length(result))
% fprintf("误差超过10的样本个数有：%d\n误差超过10的样本占总样本：%f\n", length(find(result>10)), length(find(result>10))/length(result))
%save('poolesn_nodisturb_mean.mat','suma','sumt','summean');
sumresult=sum(result);
sumtestresult=sum(result(trainLen:end));
sumvar= var(result);
sumtestvar=var(result(trainLen:end));
abovemean=length(find(result>MEAN));
aboverate=100*length(find(result>MEAN))/length(result);
%save('poolESN_with_Chen3_915_dis','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate','MEAN') 
%save('poolESN_with_house_3_1205_dis','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate','MEAN') 
MEAN = sum(result)/num;
sumresult=sum(result);
sumtestresult=sum(result(trainLen:end));
sumvar= var(result);
sumtestvar=var(result(trainLen:end));
abovemean=length(find(result>MEAN));
aboverate=100*length(find(result>MEAN))/length(result);
%save('PESN_with_allstock_912.mat','','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate') ;
%save('poolESNnew_with_finance_3_new_no055_0224.mat','','sumresult','sumtestresult','sumvar','sumtestvar','abovemean','aboverate','MEAN','MSE','X','result','yy','X','W_out') 
fact_value = yy(1:2,:);
forecast_value = yy(3:4,:);
forecast_value = reshape(forecast_value, [], 1);
fact_value = reshape(fact_value, [], 1);

reshape_data = reshape(data, [], 1);
reshape_forecast_data = data;
reshape_forecast_data(end-1:end, :) = yy(end-1:end, :);
reshape_forecast_data = reshape(reshape_forecast_data, [], 1);

figure
plot(reshape_forecast_data, 'b--', 'DisplayName', 'Forecast data of PESN', 'LineWidth', 1.5)
hold on
plot(reshape_data, 'r', 'DisplayName', 'Real data')
xlabel("number of sample")
ylabel("value")
legend
hold off

figure
plot(115400:129500, reshape_forecast_data(115400:129500), 'b--', 'DisplayName', 'Forecast data of PESN', 'LineWidth', 1.5)
hold on
plot(115400:129500, reshape_data(115400:129500), 'r', 'DisplayName', 'Real data')
xlabel("number of sample")
ylabel("value")
legend
hold off

figure
plot(792500:801600, reshape_forecast_data(792500:801600), 'b--', 'DisplayName', 'Forecast data of PESN', 'LineWidth', 1.5)
hold on
plot(792500:801600, reshape_data(792500:801600), 'r', 'DisplayName', 'Real data')
xlabel("number of sample")
ylabel("value")
legend
hold off

figure
plot([trainLen*2,trainLen*2], [0,1], 'm', 'DisplayName', "The line between training and forecasting")
hold on
plot(forecast_value, 'b--', 'DisplayName', "Forecast data of PESN", 'LineWidth', 1.5)
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