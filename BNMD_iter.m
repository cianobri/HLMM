function [X] = BNMD_iter(S, D, iters)
%% performs beta-NMD with beta = 0.5 and fixed dictionary
% stops on convergence criteria
% Input
% S - spectrogram
% D - dictionary
% Output 
% X - coefficient matrix


max_iter = iters;
err = 0.005;
iter = 0;
converged = 0;

%% initialise

X =  0.01 * ones(size(D' * S));
R = D * X;
cost = get_cost(S, R);


%% Perform optimisation

while ~converged && iter < max_iter
    
    iter = iter + 1;
    
    RR = 1./sqrt(R);
    RRR = RR .* (S./R);
           
    X = X .* (D'*RRR) ./ (D'*RR);
    
    % low rank approximation:
%     [u,s,v] = svd(X);
%     s(s < thresh) = 0;
%     X = u*s*v';
      R = D * X;
    
    %% check for convergence
    
    if mod(iter, 5) == 0
        new_cost = get_cost(S, R);
        if (cost-new_cost) / cost < err && iter > 10
           converged = 1;
        end
        cost = new_cost;
    end
    
end

end
    


function cost = get_cost(Data, R)

A = sqrt(Data);
B = sqrt(R);
C = A-B;
C = C.^2;
cost = 2 * sum(sum(C ./ B));

end













%{

%% alternative version
function grad = get_grad(C, block_AtA, groups, lambda,p)

    % does perp, 1 norm for alpha + beta = 0.5
    
   
    grad = zeros(size(C));
    
    
    for i = 1:length(groups)
        
        gg = groups{i};
        proj =  block_AtA(gg, gg) * C(gg,:);
        Cc = C(gg,:) .* proj;
        sum_CC = sqrt(sum(Cc));     % group coef
        
        f_CC = sum_CC.^(2-p);
        CCC = repmat(f_CC , length(gg), 1);
        grad(gg,:) = lambda * p * (proj ./ (CCC + eps));
    end

    %grad = lambda * (Cc ./ (CCC + eps));
    
end
%}




%{

function cost = get_cost(Data, R, X, lambda, p, iter, block_AtA, groups)

alpha = 1;
beta = -0.7;

RR = sqrt(R);

A = Data./RR;
B = - 2 * sqrt(Data);
C = RR;
D = 2 * (A+B+C);
D = sum(sum(D));


Cc = block_AtA * X;
CC = X .* Cc;

sum_CC = zeros(88,1292);

for i = 1:length(groups)
    sum_CC(i,:) = sum(CC(groups{i},:));
end
sum_CC = sqrt(sum_CC); 

E = sum(sum(sum_CC.^p)) * lambda/p;
cost = D + E;

disp([int2str(iter) '   ' num2str(D)  '   ' num2str(E) '   ' num2str(cost)]);

cost = E;

    
end

%}


%{
function grad = get_grad2(C, groups, lambda, p)

    % does 2, 1 norm for alpha + beta = 0.5
    
    tic
    
    CC = C .* C;
    
    toc
   
    for i = 1:length(groups)
        gg = groups{i};
        
        sum_CC = sqrt(sum(CC(gg,:)));
        f_CC =  sum_CC.^(2-p);
        CCC(gg,:) = repmat(f_CC , length(gg), 1);
    end

    tic
    grad = (lambda * C ./ (CCC + eps));
    toc
    
    
end
%}


