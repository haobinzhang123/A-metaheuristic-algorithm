function [Group_Best_Pos,Group_Best_Score,WAR_curve,runtime]=BEO(fhd,dim,N,Max_iteration,lb,ub,varargin)
%% parameter setting
ub = ub.*ones(dim,1);    
lb = lb.*ones(dim,1);  
R=N; 
T=Max_iteration;    
rand('twister',mod(floor(now*8640000),2^31-1))
%% initialization
WAR_curve=zeros(1,R);
x=initialization(R,dim,ub,lb);
for i=1:R
    Fitness1(i)=feval(fhd,x(:,i),varargin{:});  
end
[F_x,index]=sort(Fitness1);
A_Pos=x(:,index(1));
A_Score=F_x(1);  
t=1;  
L=0;
A=A_Score;
IT=0;
tic
%% Start iterative updates
while t<T
    %% stalking
    ub1=ub;
    lb1=lb;
    [~, max_index] = max(abs(A_Pos)); 
    lb1(max_index)=(ub(max_index)+lb(max_index))/2;
    for j=1:R
        r1=rand(dim,1);
        D = max(norm(ub-A_Pos),norm(A_Pos-lb)); 
        alpha = exp(-(norm( x(:,j)-A_Pos)/D));
        r2=rand;
        r3=tent_map(rand(1), 0.5);
        x_r=(exp(-0.2*(t/T)))*(lb1+r1.*(ub1-lb1));
        rx = floor(R*rand()+1);
        x_k= x(:,rx);
        xnew1(:,j)=x(:,j);
        xnew21(:,j)=x_r+alpha*r2*(r3*A_Pos-x_k);%盯梢
        xnew22(:,j)= lb+ub - xnew21(:,j);
    end
    xnew=[xnew1,xnew21,xnew22];
     for i=1:3*R
        ub_flag=xnew(:,i)>=ub;
        lb_flag=xnew(:,i)<=lb;
        xnew(:,i)=xnew(:,i).*(~(ub_flag+lb_flag))+(2*ub-xnew(:,i)).*ub_flag+(2*lb-xnew(:,i)).*lb_flag;
    end
    for i=1:3*R
        f(i)=feval(fhd,xnew(:,i),varargin{:});
    end
    [fx,indexf]=sort(f);
    A_Pos1=xnew(:,indexf(1));
    A_Score1=fx(1);
    if A_Score1<A_Score
        A_Score=A_Score1;
        A_Pos=A_Pos1;
    end
    x=xnew(:,indexf(1:R));
    Fitness1=fx(1:R);

    %% warning mechanism
    x0=zeros(dim,R);
    for i=1:dim
        if A_Pos(i)>=0.8*ub(i)||A_Pos(i)<=0.8*lb(i)
            center = (ub + lb) / 2;
            distance = norm(A_Pos(i) - center(i));
            max_distance=norm(ub(i));
            min_distance=0.8*norm(ub(i));
            distance_normalized=(distance-min_distance)/(max_distance-min_distance);
            lambda=distance_normalized*(5-15)+15;
            distances = sqrt((x(i,:) - A_Pos(i)).^2);
            [~, sorted_indices1] = sort(distances);
            xnew3(i,:) = zeros(1,R);
            poissrn_values = poissrnd(lambda, 1, R);
            
            for j = 1:R
                xnew3(i,j) = A_Pos(i) + (x(i,sorted_indices1(j)) - A_Pos(i)) .* (1 + poissrn_values(j));
            end
            lambda_normalized=(lambda-5)/(15-5);
            bate=sin(lambda_normalized*pi/2)*(0.953-0.89)+0.89;
            xnew3(i,:)=xnew3(i,:)+bate*(A_Pos(i)-xnew3(i,:));
            x0(i,:)=xnew3(i,:);
        end
    end
    %% hovering
    for j=1:R
        rotation_angle = rand*2*pi;
        rotation_matrix = eye(dim);
        for ri = 1:dim-1
            rotation_matrix(ri,ri) = cos(rotation_angle);
            rotation_matrix(ri,ri+1) = -sin(rotation_angle);
            rotation_matrix(ri+1,ri) = sin(rotation_angle);
            rotation_matrix(ri+1,ri+1) = cos(rotation_angle);
        end
        xnew4(:,j) = A_Pos +rotation_matrix * (x(:,j) - A_Pos);
    end
    for i=1:R
        ub_flag=xnew4(:,i)>=ub;
        lb_flag=xnew4(:,i)<=lb;
        xnew4(:,i)=xnew4(:,i).*(~(ub_flag+lb_flag))+(2*ub-xnew4(:,i)).*ub_flag+(2*lb-xnew4(:,i)).*lb_flag;
        f=feval(fhd,xnew4(:,i),varargin{:});
        if f<Fitness1(i)||f<mean(Fitness1)
            Fitness1(i)=f;
            x(:,i)= xnew4(:,i);
        end
    end
               
    [F_x,index]=sort(Fitness1);
    A_Pos1=x(:,index(1));
    A_Score1=F_x(1);
    if A_Score1<A_Score
        A_Score=A_Score1;
        A_Pos=A_Pos1;
    end
    %% catching  
        d1 = sqrt(sum((x - mean(x)).^2, 1));
        md1=(median(d1)+mean(d1))/2;
        for j=1:R
            xnew50(:,j)=2*x(:,j)-A_Pos +randn(dim,1) * md1;
        end
        d2 = sqrt(sum((xnew50 - mean(xnew50)).^2, 1));
        md15=(median(d2)+mean(d2))/2;
        md2=1-md1/md15;
        for j=1:R
          xnew5(:,j)=xnew50(:,j)+md2*(A_Pos-xnew50(:,j));
        end  
    xnew=zeros(dim,R);
    for i=1:dim
        flag1=sum((x0(i,:)).^2)>0;
        xnew(i,:)=xnew5(i,:).*(~flag1)+x0(i,:);
    end
    for i=1:R
        ub_flag=xnew(:,i)>=ub;
        lb_flag=xnew(:,i)<=lb;
        xnew(:,i)=xnew(:,i).*(~(ub_flag+lb_flag))+(lb+rand(dim,1).*(ub-lb)).*ub_flag+(lb+rand(dim,1).*(ub-lb)).*lb_flag;
        f=feval(fhd,xnew(:,i),varargin{:});
        if f<Fitness1(i)
            Fitness1(i)=f;
            x(:,i)= xnew(:,i);
        end
    end

    [F_x,index]=sort(Fitness1);
    A_Pos1=x(:,index(1));
    A_Score1=F_x(1);
    if A_Score1<A_Score
        A_Score=A_Score1;
        A_Pos=A_Pos1;
    end
    for j=1:R
        xnew6(:,j)= A_Pos+exp(randn(dim,1)/pi).*( A_Pos - x(:,j));
    end
    for i=1:R
        ub_flag=xnew6(:,i)>ub;
        lb_flag=xnew6(:,i)<lb;
        xnew6(:,i)=xnew6(:,i).*(~(ub_flag+lb_flag))+(2*ub-xnew6(:,i)).*ub_flag+(2*lb-xnew6(:,i)).*lb_flag;
        f=feval(fhd,xnew6(:,i),varargin{:});
        if f<Fitness1(i)||f<mean(Fitness1)
            Fitness1(i)=f;
            x(:,i)= xnew6(:,i);
        end
    end
    [F_x,index]=sort(Fitness1);
    A_Pos1=x(:,index(1));
    A_Score1=F_x(1);
    if A_Score1<A_Score
        A_Score=A_Score1;
        A_Pos=A_Pos1;
    end
    %% migration
    if t>=0.1*T||L>0.1*T/2
        for j=1:R
            Fitness(j)=feval(fhd,x(:,j),varargin{:});
            z=1/(2*exp(-(A_Score/(Fitness(j)+eps))));
            s=-ones(dim,1);
            s0=s+2*rand(dim,1);
            xnew7(:,j)=A_Pos+z.*s0.*(x(:,j)-(0.4+0.6.*tent_map(rand(dim,1), 0.5)).*A_Pos);
        end
    for i=1:R
        ub_flag=xnew7(:,i)>ub;
        lb_flag=xnew7(:,i)<lb;
        xnew7(:,i)=xnew7(:,i).*(~(ub_flag+lb_flag))+(2*ub-xnew7(:,i)).*ub_flag+(2*lb-xnew7(:,i)).*lb_flag;
        f=feval(fhd,xnew7(:,i),varargin{:});
        if f<Fitness1(i)
            Fitness1(i)=f;
            x(:,i)= xnew7(:,i);
        end
    end
    [F_x,index]=sort(Fitness1);
    A_Pos1=x(:,index(1));
    A_Score1=F_x(1);
    if A_Score1<A_Score
        A_Score=A_Score1;
        A_Pos=A_Pos1;
    end

    end
    %% courtship
    if t>=0.3*T||L>0.3*T/2
                  f = 1./(1*(1+exp(-(14*t/T-9))))+0.3;%0.3-1.3
        for j=1:R
            if mod(j,2)==1
                r5=tent_map(rand(1), 0.5);
                
                xnew8(:,j)=x(:,j)+r5*f*(A_Pos-x(:,j))+rand(dim,1).*cos((pi*j)/R).*(A_Pos-x(:,j));
            else
                r6=tent_map(rand(1), 0.5);

                xnew8(:,j)=x(:,j)+r6*f*(A_Pos-x(:,j))+randn(dim,1).*sin((pi*j)/R).*(A_Pos-x(:,j));
            end
        end
        
        for i=1:R
            ub_flag=xnew8(:,i)>ub;
            lb_flag=xnew8(:,i)<lb;
            xnew8(:,i)=xnew8(:,i).*(~(ub_flag+lb_flag))+(2*ub-xnew8(:,i)).*ub_flag+(2*lb-xnew8(:,i)).*lb_flag;
            f=feval(fhd,xnew8(:,i),varargin{:});
            if f<Fitness1(i)
                Fitness1(i)=f;
                x(:,i)= xnew8(:,i);
            end
        end
        [F_x,index]=sort(Fitness1);
        A_Pos1=x(:,index(1));
        A_Score1=F_x(1);
        if A_Score1<A_Score
            A_Score=A_Score1;
            A_Pos=A_Pos1;
        end

        %% hatching
        distances1 = sqrt(sum((x - A_Pos).^2, 1));
        [~, sorted_indices1] = sort(distances1);
        xnew9 = zeros(size(x));
        thet =2 ;
        miu = 0.2*sin(rand*2*pi);
        Q = thet * randn(size(x, 1), size(x, 2)) + miu;
        for i = 1:size(x, 2)
            xnew9(:,i) = A_Pos + (x(:,sorted_indices1(i)) - A_Pos) .* (1 + Q(:,i));
        end
 
        for i=1:R
            ub_flag=xnew9(:,i)>ub;
            lb_flag=xnew9(:,i)<lb;
            xnew9(:,i)=xnew9(:,i).*(~(ub_flag+lb_flag))+(2*ub-xnew9(:,i)).*ub_flag+(2*lb-xnew9(:,i)).*lb_flag;
            f=feval(fhd,xnew9(:,i),varargin{:});
            if f<Fitness1(i)||f<mean(Fitness1)
                Fitness1(i)=f;
                x(:,i)= xnew9(:,i);
            end
        end
        [F_x,index]=sort(Fitness1);
        A_Pos1=x(:,index(1));
        A_Score1=F_x(1);
        if A_Score1<A_Score
            A_Score=A_Score1;
            A_Pos=A_Pos1;
        end

        
    end
    A=[A,A_Score];
    if t>2
        if A(t)==A(t-1)
            L=L+1;
        end
    end


    Group_Best_Score=A_Score;
     disp( ['best score',num2str(Group_Best_Score)])
    Group_Best_Pos=A_Pos;
    t=t+1;  
    WAR_curve(t)=Group_Best_Score;
    
    
end
runtime=toc;

%%
 function P=initialization(R,dim,ub,lb)
  for a = 1:R
    for b = 1:dim
        Rand=tent_map(rand(1), 0.4999);
        P(b, a) = lb(b) + Rand * (ub(b) - lb(b));
    end
  end
end
    function output = tent_map(x, a)
        if x < a
            output = x / a;
        else
            output = (1 - x) / (1 - a);
        end
    end


end