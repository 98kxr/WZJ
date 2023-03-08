function [Phi, Length] = PartHadamardMtx(M, N, Loc)
%PartHadamardMtx Summary of this function goes here  
%   Generate part Hadamard matrix   
%   M -- RowNumber  
%   N -- ColumnNumber
%   Loc -- 从哈达玛矩阵中取出指定的行和列。其中，Loc第一个元胞是行索引，第二个元胞为列索引。
%   Phi -- The part Hadamard matrix
%   Length -- 补全后的哈达玛矩阵的大小，适用于取指定行和列的情况
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
%L就是最小满足条件的行数（=列数）。

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