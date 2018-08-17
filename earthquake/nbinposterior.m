function [RP] = nbinposterior(y,N)
% Sample from the negative binomial parameter posterior using
% Metropolis-hastings
% Andreas Svensson 2018

burn_in = 100;
thinning = 50;

K = burn_in + (N-1)*thinning;

RP_full = zeros(K,2);
p = zeros(K,1);

try
    RP_full(1,:) = nbinfit(y);
catch
    RP_full(1,:) = [3 .5];
end

if RP_full(1,1)==inf
    RP_full(1,:) = [3 .5];
end

p(1) = prod(nbinpdf(y,RP_full(1,1),RP_full(1,2)));

prop_var = .1*eye(2);
ch_prop_var = chol(prop_var);
for k = 2:K
    cand = RP_full(k-1,:) + randn([1 2])*ch_prop_var;
    candp = prod(nbinpdf(y,cand(1),cand(2)));
    
    if rand < candp/p(k-1)
        % accept
        RP_full(k,:) = cand;
        p(k) = candp;
    else
        % reject
        RP_full(k,:) = RP_full(k-1,:);
        p(k) = p(k-1);
    end
end

RP = RP_full(burn_in:thinning:K,:);


end

