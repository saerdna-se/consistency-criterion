% Source code for the time-series example in
% "Data Consistency Approach to Model Validation",
% Andreas Svensson, Dave Zachariah, Petre Stoica, Thomas B. Schön
%
% Code Andreas Svensson 2018
clear, clf
rng(1)

    clf
    nv = [10 100 1000];
    MC = 100;
    N = 50;
    Mp = 200;
    M = 200;
    sigma2 = 1;

    lnpdf = @(y,theta,sigma2) -1/2*log(2*pi*sigma2) - 1/(2*sigma2)*(y(2:end)-theta*y(1:end-1)).^2;
    PFAs = zeros(MC,length(nv));
    for in = 1:3
        n = nv(in);
        for mc = 1:MC
            
            % simulate data from saturated AR model
            ygiv = zeros(1,n);
            for ni = 2:n
                ygiv(ni) = max(0.7*ygiv(ni-1)+randn*sigma2,-.3);
            end
            m_0 = 0;
            ell = 0;
            lambda = 0;
            v = inf;

            % sample parameters
            sigma2s = 1./gamrnd(((n-1+ell)-3)/2,2/(lambda + sum(ygiv(2:end).^2) + m_0^2/v - (m_0/v + sum(ygiv(2:end).*ygiv(1:end-1))^2/(sum(ygiv(1:end-1).^2)+1/v))),[N,1]);
            post_mean = (m_0/v+ sum(ygiv(1:n-1).*ygiv(2:n)))/(1/v+sum(ygiv(1:end-1).^2));
            thetas = (randn([N,1]).*sqrt(sigma2s/(1/v+sum(ygiv(1:end-1).^2))) + post_mean);
            
            % simulate new data and compute PFA
            Tgiv = zeros(N,1);
            Tsim = zeros(N,M);
            PFA_o = zeros(N,1);
            for j = 1:N
                ysimp = zeros(n,Mp);
                zsimp = zeros(n-1,Mp);
                for k = 1:Mp
                    ysimp(:,k) = simmod(thetas(j),sigma2s(j),n);
                    zsimp(:,k) = lnpdf(ysimp(:,k),thetas(j),sigma2s(j));
                end
                Ez = mean(zsimp,2)';
                Vz = var(zsimp,[],2)';
                ysim = zeros(n,Mp);
                zsim = zeros(n-1,Mp);
                for k = 1:Mp
                    ysim(:,k) = simmod(thetas(j),sigma2s(j),n);
                    zsim(:,k) = lnpdf(ysim(:,k),thetas(j),sigma2s(j));
                end
                zgiv = lnpdf(ygiv,thetas(j),sigma2s(j));
                Tgiv(j) = mean((zgiv-Ez).^2./Vz);
                Tsim(j,:) = mean((zsim-Ez(ones(1,M),:)').^2./Vz(ones(1,M),:)',1)';
                PFA_o(j) = mean(Tsim(j,:)<Tgiv(j));
            end
            PFAs(mc,in) = min(mean(PFA_o),1-mean(PFA_o));
        end
        subplot(4,1,in)
        histogram(PFAs(:,in),n,'BinLimits',[0 1],'NumBins',20)
        title(['n = ',num2str(n)])
        drawnow
    end

function [ysim] = simmod(theta,sigma2,n)
ysim = zeros(1,n);
for ni = 2:n
    ysim(ni) = theta*ysim(ni-1) + sqrt(sigma2)*randn;
end
end