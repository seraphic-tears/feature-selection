% LapScore
clear
clc
load('./datasets/Isolet.mat');
 NewFeaNum=[50 100 150 200 250 300];
% NewFeaNum=[ 200 ];
for i=1:6
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 8;
    % options.WeightMode = 'Cosine';
    W = constructW(fea,options);
    s = LaplacianScore(fea,W);
    [sorted_s,ind]=sort(s,'descend');
    Newfea=fea(:,ind(1:NewFeaNum(i)));
    label=litekmeans(Newfea,26,'Replicates',20);
    [Acc(i),Nmi(i),~]=ClusteringMeasure(gnd,label);
end

 %spec
 clear
clc
load('./datasets/Isolet.mat');
            X = fea';%dim*num
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
            [ wFeat, SF ] = fsSpectrum( A0, fea, -1)
             [dumb idx] = sort(wFeat,'descend');
           NewFeaNum=[50 100 150 200 250 300];

            for i=1:6
                Newfea=X(idx(1:NewFeaNum(i)),:)';
                label=litekmeans(Newfea,26,'Replicates',20);
                [Acc(i),Nmi(i),~]=ClusteringMeasure(gnd,label);
            end

      % MCFS
      clear
      clc
      load('./datasets/Isolet.mat');
    NewFeaNum=[50 100 150 200 250 300];
      [m, n]=size(fea);
      for i=1:6
          options = [];
          options.nUseEigenfunction = 5;
          FeaNumCandi =n ;
          [index,~,score] = MCFS(fea,NewFeaNum(i),options);
          s=cell2mat(score);
          [sorted_s,ind]=sort(s,'descend');
          mcfs_ind=cell2mat(index)
          Newfea=fea(:,mcfs_ind);
          label=litekmeans(Newfea,26,'Replicates',20);
          [Acc(i),Nmi(i),~]=ClusteringMeasure(gnd,label);
      end
   

       
      %  UDFS
      clear
      clc
      load('./datasets/Isolet.mat');
      NewFeaNum=[50 100 150 200 250 300];
          
      for i=1:6
          [~,sorted_score,ind] = UDFS(fea,26)
          Newfea=fea(:,ind(1:NewFeaNum(i)));
          label=litekmeans(Newfea,26,'Replicates',20);
          [Acc(i),Nmi(i),~]=ClusteringMeasure(gnd,label);
      end
      
      
             
                      % llcfs
        clear
        clc
        load('./datasets/Isolet.mat');
       NewFeaNum=[50 100 150 200 250 300];
         
        [m, n]=size(fea);
        for i=1:6
            ind = llcfs(fea)
            Newfea=fea(:,ind(1:NewFeaNum(i)));
            label=litekmeans(Newfea,26,'Replicates',20);
            [Acc(i),Nmi(i),~]=ClusteringMeasure(gnd,label);
        end      
      
     
           
            % RuFS
            clear
            clc
            load('./datasets/isolet.mat');
            options.nu = 5;
            options.alpha  =5;
            options.beta = 4;
            options.MaxIter = 10;
            options.epsilon = 1e-4;
            options.verbose = 1;
            
            X = fea';%dim*num
            c = 26;
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
           NewFeaNum=[50 100 150 200 250 300];
     
            for i=1:6
                Newfea=X(idx(1:NewFeaNum(i)),:)';
                label=litekmeans(Newfea,26,'Replicates',20);
                [Acc(i),Nmi(i),~]=ClusteringMeasure(gnd,label);
            end
         
           
                 
     %      SOCFS method        
             
     clear
     clc
     load('./datasets/Webkb_texas.mat');
     NewFeaNum=[50 100 150 200 250 300];
     ind=SOCFS(fea',200,100,100,50,50)
     clusternum=7;
     for i=1:6
         Newfea=fea(:,ind(1:NewFeaNum(i)));
         for iter=1:20
             label=litekmeans(Newfea,clusternum,'Replicates',20);
             [Acc(iter),NMI(iter)]=ClusteringMeasure(gnd,label);
         end
         meanacc(i)=mean(Acc);
         meannmi(i)=mean(NMI);
     end
                 
           %SOGFS       
                 
           clc
           load('./datasets/YALE.mat');
           NewFeaNum=[50 100 150 200 250 300];
           ind=SOGFS(fea',1000,10,15,5);
           clusternum=15;
           for i=1:6
               Newfea=fea(:,ind(1:NewFeaNum(i)));
               for iter=1:20
                   label=litekmeans(Newfea,clusternum,'Replicates',20);
                   [Acc(iter),NMI(iter)]=ClusteringMeasure(gnd,label);
               end
               meanacc(i)=min(Acc);
               meannmi(i)=min(NMI);
           end
                 
                 
            % jelsr
            
            load('./datasets/Webkb_texas.mat');
            data = fea;
            k=5;
            clusternum=7;
            Reducednum=400;
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
                        infind=find(A>10);
                        A(infind)=1;

            [W_compute, Y, obj] = jelsr(data, A,  Reducednum,100,10);
             [dumb ind] = sort(sum(W_compute.*W_compute,2),'descend');
            NewFeaNum= [50 100 150 200 250 300];
       
                 for i=1:6
                     Newfea=fea(:,ind(1:NewFeaNum(i)));
                     for iter=1:20  
                     label=litekmeans(Newfea,clusternum,'Replicates',20);
                     [Acc(iter),NMI(iter)]=ClusteringMeasure(gnd,label);
                     end     
               meanacc(i)=mean(Acc);
               meannmi(i)=mean(NMI);
                 end 
                 
                 

                 
      
            
          % my method 
           
            clear
            clc
            load('./datasets/COIL20.mat');
          
            clusternum=20;
            NewFeaNum=200 ;
                 ind = pkddfs(fea,10,100,1000,clusternum)% c: number of clusters
                 for i=1:1
                     Newfea=fea(:,ind(1:NewFeaNum(i)));
                     label=litekmeans(Newfea,clusternum,'Replicates',20);
                     [Acc(i),NMI(i)]=ClusteringMeasure(gnd,label);
                 end
                 
                 
                 
                    clear
            clc
            load('./datasets/jaffe.mat');
          
            clusternum=10;
            NewFeaNum=[ 200 ];
            
                 [ind,objour] = pkddfs(fea,gnd,10,100,1000,clusternum)% c: number of clusters
                 for i=1:1
                     Newfea=fea(:,ind(1:NewFeaNum(i)));
                     label=litekmeans(Newfea,clusternum,'Replicates',20);
                     [Acc(i),NMI(i)]=ClusteringMeasure(gnd,label);
                 end
           
           
             %NDFS method
             clear
      clc
      load('./datasets/smallusps.mat');
            X = fea';%dim*num
            c = 10;
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
           aa = 0.1;
           bb = 1;
           cc = 1000;
           [tY,Wa,~] = AAAI2012(X,full(L1),tLabel,W,100,aa,bb,cc);
           [dumb idx] = sort(sum(Wa.*Wa,2),'descend');
           
            NewFeaNum=[50 100 150 200 250 300];
            for i=1:6
                Newfea=X(idx(1:NewFeaNum(i)),:)';
                label=litekmeans(Newfea,10,'Replicates',20);
                [Acc(i),Nmi(i),~]=ClusteringMeasure(gnd,label);
            end
           
           
           
                                             
       
      