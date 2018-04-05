function [ind]= NMSFS(X,a,b,gamma,k,Q )
%  X: data matrix, every column denotes a sample.
%  a: regression parameter.
%  b:  L2,1-norm parameter.
%  gamma: a large enough number which guarantees orthogonal condition.
%  k: clusternum.
%  Q: a nonnegative and orthogonal matrix.
[dim samples]=size(X);
W = orth(rand(dim,k));
D = eye(dim);
H = eye(samples)- (1/samples)*ones(samples,1)*ones(samples,1)';
Iter=300;
err=10e+6;
count=1;
while(count<Iter)
                                              %% fixing Q, update W
    A=(1+a)*X*H*X'+b*D-X*H*Q*Q'*H*X';
    B=2*a*X*H*Q;
    W=GPI(A,0.5*B);
                                              %% fixing W, update Q
    C=a*H-H*X'*W*W'*X*H;
    E=2*a*H*X'*W;
    Q= Q.*(2*gamma*Q + eps)./(2*C*Q-E + 2*gamma*Q*Q'*Q + eps);
    Q = Q*diag(sqrt(1./(diag(Q'*Q)+eps)));
                                              %% fixing W and Q, update D
    wi = sqrt(sum(W.*W,2)+eps);
    d = 0.5./wi;
    D = diag(d);
    
    err=err-(trace(H*X'*W*W'*X*H)-trace(Q'*H*X'*W*W'*X*H*Q)+a*norm((H*(X'*W-Q)),'fro')^2+b*trace(W'*D*W));
    if(abs(err)<0.00001)
        break;
    end
    count=count+1;
end

wi = sqrt(sum(W.*W,2));
[~, ind] = sort(wi,'descend');

end


