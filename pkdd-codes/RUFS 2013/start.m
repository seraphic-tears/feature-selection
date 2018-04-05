
options.nu = 5;
options.alpha  =5;
options.beta = 4;
options.MaxIter = 10;
options.epsilon = 1e-4;
options.verbose = 1;

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

[W,F,G] = RUFS(X',L1,tLabel,options);

[dumb idx] = sort(sum(W.*W,2),'descend');

  NewFeaNum=[40 80 120 160 200 240];
     for i=1:6
Newfea=X(idx(1:NewFeaNum(i)),:)';
  label=litekmeans(Newfea,20,'Replicates',20);
  [Acc(i),Nmi(i),~]=ClusteringMeasure(gnd,label);
   end
%    X: NExample x NFea data matrix.
%    L: NExample x NExample local learning regularization matrix. Lap
%    matrix
%    G0: Initial NExample x NClus encoding matrix.cluster result
%    options: 1-by-1 structure for parameters.
%        options.nu: graph regularization parameter.
%        options.alpha: parameter on the regression term.
%        options.beta: parameter to control the row-sparsity of the 
%                      projection matrix.
%        options.MaxIter: maximal number of iterations, e.g., 10.
%        options.epsilon: convergence precision, e.g., 1e-4.
%        options.verbose: if verbose, 1 or 0.

% function [ W, F, G ] = RUFS(X, L, G0, options)