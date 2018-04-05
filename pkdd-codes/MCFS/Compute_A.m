
[n , m]=size(fea);
colmean=mean(fea,1);
centerfea=fea-repmat(colmean,n,1);

for i=1:m
    for j=1:m   
        
        B(i,j)= centerfea(:,i)'*centerfea(:,j)/(norm(centerfea(:,i))* norm(centerfea(:,j))) ;
        
    end
end
A=B.*B;



        
