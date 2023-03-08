function [Phi, Length] = PartHadamardMtx(M, N, Loc)
%PartHadamardMtx Summary of this function goes here  
%   Generate part Hadamard matrix   
%   M -- RowNumber  
%   N -- ColumnNumber
%   Loc -- �ӹ����������ȡ��ָ�����к��С����У�Loc��һ��Ԫ�������������ڶ���Ԫ��Ϊ��������
%   Phi -- The part Hadamard matrix
%   Length -- ��ȫ��Ĺ��������Ĵ�С��������ȡָ���к��е����
%% parameter initialization  
%Because the MATLAB function hadamard handles only the cases where n, n/12,or n/20 is a power of 2
flag = 0;
Lt(1) = max(M,N);
Lt(2) = Lt(1)/12;
Lt(3) = Lt(1)/20;
for i = 1:3
    if 2^(ceil(log2(Lt(i)))) == Lt(i)
        flag = 1;
        break;
    else
        continue;
    end
end

L = Lt(1);
if flag == 0
    L = 2^(ceil(log2(L)));
end

Length = L;
%L������С����������������=��������

%% Generate part Hadamard matrix     
Phi = [];  
Phi_t = hadamard(L);  

if ~exist('Loc' , 'var')
    RowIndex = randperm(L);  
    Phi_t_r = Phi_t(RowIndex(1:M),:);  
    ColIndex = randperm(L);
    Phi = Phi_t_r(:,ColIndex(1:N));
else
    Phi = Phi_t(Loc{1}, Loc{2});
end

end