X = fea';%dim*num
c = 40;
num = size(X,2);
k = 5;
distX = L2_distance_1(X,X);
%distX = sqrt(distX);
[distX1, idx] = sort(distX,2);
A = zeros(num);
for i = 1:num
    di = distX1(i,2:k+2);
    id = idx(i,2:k+2);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;

A0 = (A+A')/2;

[Label,L1,V1] = sc(A0,c);

tLabel = zeros(num,c);
for i = 1:num
    tLabel(i,Label(i)) = 1;
end;
W = zeros(size(X,1),c);
%	X: Rows of vectors of data points  %d*n
%	L: The laplacian matrix. n*n
%   F: the cluster result  %n*c
%   W: the feature selection matrix d*c

result = zeros(1250,5);
log = {};
parfor i = 0:124
    param = [0.1,0.5,1,5,10];
    aa = param(fix(i/25)+1);
    bb = param(fix(rem(i,25)/5)+1);
    cc = param(rem(rem(i,25),5)+1);
    %     a(i+1,:) = [aa,bb,cc];
    [tY,Wa,~] = AAAI2012(X,full(L1),tLabel,W,100,aa,bb,cc);
    log(i+1,:) = {Wa,aa,bb,cc};
end;

% [dumb idx] = sort(sum(W.*W,2),'descend');
% selectfeature = [5:5:50,50:50:300];
% for ii =1:16
%     tre =  ClusteringMeasure(YY,kmeans(X(idx(1:ii),:)',c));
%     
% end;
% mY = max(tY');
% for i = 1:num
%     Y(i) = find(tY(i,:)==mY(i));
% end;



% function [F,W,obj]=AAAI2012(X,L,F,W,maxIter,alpha,beta,gamma)
%	X: Rows of vectors of data points
%	L: The laplacian matrix.
%   F: the cluster result
%   W: the feature selection matrix