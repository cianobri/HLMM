function [W, H, alpha, weights, D] = LRNMM_KL_cond3(X, D, r, K, beta)
% returns local mixure, mixture activations and mixting
% coefficients of input X using dictionary D

% Inputs:
% X - input data
% D - insput dictionary
% r - rank of each local model
% K number of local models
% beta - parameter for beta-NMF
%     
% Outputs:
% W - local models
% H - local activations
% alpha - mixture weights
% weights - normalized weights
% D - updated dictionary
    
% initialize
W  = rand(size(D,2),r,K); 
lW = 0;%0.02;
lH = 0;%0.005; % sparisty param
iter = 0;
MAX_ITER = 25;
TOL      = 1e-5;
D = D + eps;

% INITIALIZATION ----------------------------------------------------------
e = round(size(X,2) / K);
c = 1;
count = 1;
while c + e < size(X,2)
    t(count) = c + e;
    count = count + 1;
    c = c+e;
end
t = t - round(e/2);
t = [1 t size(X,2)];
for i=1:K
    rn = randi(t(i+1)-t(i),1,r) + t(i);
    mu(i)     = mean(rn);
    sig(i)    = 5*size(X,2) / K;
    W(:,:,i)  = BNMD(X(:,rn), D);
    S(:,:,i)  = eye(size(X,1),size(X,1));
    H(:,:,i)  = BNMD(X, D*W(:,:,i));
end

% -------------------------------------------------------------------------
alpha = zeros(K,size(X,2));
w = zeros(K,size(X,2));
new_likelihood = getL(X,D,W,H,S,w,lW,lH);
old_likelihood = new_likelihood;
weights = ones(size(W,3),1)*(1/size(W,3));
while 1
    
    iter = iter + 1;
    
    disp(['Likelihood: ' num2str(old_likelihood)]);
    
    if new_likelihood < old_likelihood && iter > 1
        %break
    end
    
    % E-step:
    total = zeros(1,size(X,2));

    alpha = zeros(size(W,3),size(X,2));
    Xt = normc(X);
    for k = 1:size(W,3)
        rec = D*W(:,:,k)*H(:,:,k);
        rec = normc(rec);
        err = 1/(beta*(beta-1));
        err = err*(Xt.^beta + (beta-1)*rec.^beta - beta*(Xt.*(rec.^(beta-1))));
        err = - err;
        err = sum(err,1) + log(weights(k));
        alpha(k,:) = err;
    end

    for jj = 1:size(alpha,2)
        a = alpha(:,jj);
        [~,id] = sort(a);
        a(setdiff(1:length(a),id(end-4:end))) = 0;
        %a = a / sum(a);
        alpha(:,jj) = a;
    end
    alpha = alpha ./ repmat(sum(alpha,1),size(alpha,1),1);
    weights = (1/size(alpha,2))*sum(alpha,2);
  
    D  = normc(D);

    for i=1:K
        A = diag(alpha(i,:));
        Wt = update_e(X*A,D,H(:,:,i)*A, 5, W(:,:,i), beta, 0.5);
        W(:,:,i) = Wt;
        Ht = BNMD_var(X, D*Wt, beta);
        H(:,:,i) = Ht;
        
        ao = alpha(i,:)*ones(size(X,2),1);

        [~, ss] = update_mix_params(size(X,2), alpha(i,:), mu(i));
        sig(i) = ss;%max(ss,20);
        
    end

    old_likelihood  = new_likelihood;
    new_likelihood  = getL(X,D,W,H,S,w,lW,lH);

    if abs(old_likelihood - new_likelihood) < TOL
        disp(['Likelihood: ' num2str(new_likelihood)]);
        break
    end
    if iter == MAX_ITER
        break
    end
end
end

function [L] = getL(X,D,W,H,S,w,lW,lH)
% calculate model likelihood
L = 0;

for i = 1:size(W,3)
    
    L = L + sum(sum(poisson_pdf(X, D*W(:,:,i)*H(:,:,i))*diag(w(i,:))));
    
end

tW = lW*sum(abs(W)); tW = sum(tW(:));
tH = lH*sum(abs(H)); tH = sum(tH(:));
L = sum(log(L)) - tW - tH;

end

function [x] = poisson_pdf(z, mu)
% pdf
x = (mu.*z).*(exp(-mu));
end


function [Y1, Y2] = update_param(X, D, Wt, Ht, S, alpha, w, lW, lH, beta)

TOL      = 1e-5;
MAX_ITER = 50;
diff = 2*TOL;
iter = 0;
indh = 0;
indw = 0;
while diff > TOL
    iter = iter + 1;
    
    oa = ones(size(X,1),1)*alpha;
    ao = alpha*ones(size(X,2),1);
    
    A    = diag(alpha);

    Wtp = Wt;
    Htp = Ht;
    
    d1 = ((D*Wt*Ht*A).^(-1).*X);
    d2 = (ones(size(X)));
    
    if 0.5/(numel(Wt))*norm(Wtp - Wt)^2 > TOL
        Gw = (D'*d1*(Ht*A)') ./ (D'*d2*(Ht*A)' + lW);
        Wt = Wt.*(Gw);
    else
        indw = 1;
    end
    
    if 0.5/(numel(Ht))*norm(Htp - Ht)^2 > TOL
        Gh = ((D*Wt)'*d1*A') ./ ((D*Wt)'*d2*A' + lH);
        Ht = Ht.*(Gh);
    else
        indh = 1;
    end
    
    diff = 0.5*((0.5/(numel(Ht)))*norm(Htp - Ht)^2 + (0.5/(numel(Wt)))*norm(Wtp - Wt)^2);
    
    if iter == MAX_ITER
        %disp('MAX ITER')
        break
    end
    if indh == 1 && indw == 1
        %disp('TOL')
        break
    end
end
Y1 = Wt;
Y2 = Ht;
end


function [D] = update_dict(X, W, H, D, alpha, S, beta)

TOL      = 1e-5;
MAX_ITER = 5;
diff = 2*TOL;
iter = 0;

while diff > TOL
    iter = iter + 1;
    
    V = zeros(size(D));
    Vt = V;

    for i = 1:size(W,3)

        oa = ones(size(X,1),1)*alpha(i,:);
        ao = alpha(i,:)*ones(size(X,2),1);
        
        Wt = W(:,:,i);
        Ht = H(:,:,i);
        
        A    = diag(alpha(i,:));
        dwha = D*Wt*Ht*A;
        V    = V + (((dwha).^(beta-2) .* X)*(Wt*Ht*A)') ./ (ones(size(X))*(Wt*Ht*A)');
        Vt   = Vt + BNMD_var((X*A)', (Wt*Ht*A)', beta)';
    end
    Dp = D;
    D  = D .* (Vt);
    %D = Vt;
    diff = (1/(numel(D)))*norm(D - Dp)^2;
    
    if iter == MAX_ITER
        break
    end
	if diff < TOL
        %disp('TOL')
        break
    end
end
end

function [x] = normal_pdf(z, mu, s)
% pdf
d = size(z,1);
x = exp((-0.5*(z-mu)'*inv(s)*(z-mu)) / (sqrt(2*pi*det(s))));
end

function [m,s] = update_mix_params(J, alpha, m)
%m = (1/sum(alpha))*alpha*([1:J]');
s = (1/sum(alpha))*alpha*(([1:J] - ones(1,J)*m).^2)';
%s = 1;
end