 load('sonar.mat')
 data = sonar;
 k=5;
 num = size(data',2);
dim = size(data',1);
 distX = L2_distance_1(data',data');
[distX1, idx] = sort(distX,2);
A = zeros(num);
for i = 1:num
    di = distX1(i,2:k+2);
    id = idx(i,2:k+2);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;
[W_compute, Y, obj] = jelsr(data, A, 10,10,5);

a = W_compute.^2;
[~,s] = sort(sum(a,2));
new = data(:,s(1:5));
result_kmeans = ClusteringMeasure(R1,kmeans(new,2));