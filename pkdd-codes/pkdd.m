function [Newind,objvalue]= pkddfs(X,Y,a,b,gamma,k )
%
% intialization
X=X';
% gamma=10e+6;
[dim samples]=size(X);
W = orth(rand(dim,k));
% Q = orth(rand(samples,k));
temp=unique(Y);
clusternum=length(temp);
%if clusternum==k
ori_Q = zeros(samples,clusternum); 
for i=1:samples
    ori_Q(i,Y(i))=1;
end
% Q = ori_Q*diag(sqrt(1./(diag(ori_Q'*ori_Q)+eps)));
Q =rand(samples, clusternum);
D = eye(dim);
H = eye(samples)- (1/samples)*ones(samples,1)*ones(samples,1)';
Iter=150;
err=10e+6;
count=1;
objvalue=1e+4;
while(count<Iter)
% % fixing Q, update W
A=(1+a)*X*H*X'+b*D-X*H*Q*Q'*H*X';
B=2*a*X*H*Q;
W=GPI(A,0.5*B);
% % fixing W, update Q
C=a*H-H*X'*W*W'*X*H;
E=2*a*H*X'*W;
% Q=GPI(C,0.5*E);
    Q = Q*diag(sqrt(1./(diag(Q'*Q)+eps)));
    Q= Q.*(2*gamma*Q + eps)./(2*C*Q-E + 2*gamma*Q*Q'*Q + eps);
    Q = Q*diag(sqrt(1./(diag(Q'*Q)+eps)));
% % fixing W and Q, update D
  wi = sqrt(sum(W.*W,2)+eps);
  d = 0.5./wi;
  D = diag(d);
  
  err=err-(trace(H*X'*W*W'*X*H)-trace(Q'*H*X'*W*W'*X*H*Q)+a*norm((H*(X'*W-Q)),'fro')^2+b*trace(W'*D*W));
      if(abs(err)<0.0001)
          break;
      end
  count=count+1;
%   objvalue(count)=trace(H*X'*W*W'*X*H)-trace(Q'*H*X'*W*W'*X*H*Q)+a*norm((H*(X'*W-Q)),'fro')^2+b*trace(W'*D*W);
 objvalue(count)=trace(H*X'*W*W'*X*H)-trace(Q'*H*X'*W*W'*X*H*Q)+a*norm((H*(X'*W-Q)),'fro')^2+b*sum(sqrt(sum(W.*W,2)));
end

    wi = sqrt(sum(W.*W,2));
    [temp, ind] = sort(wi,'descend');
    Newind=ind;
    save temp
end


