% Source code for the Gaussian model example in
% "Data Consistency Approach to Model Validation",
% Andreas Svensson, Dave Zachariah, Petre Stoica, Thomas B. Schön
%
% Code Andreas Svensson 2018

clear, clf

% case 1: fix, correct model (fig 4a, left)
% case 2: fix, wrong model (fig 4b, left)
% case 3: correct model class (fig 4a, right)
% case 4: wrong model class (fig 4a, right)

for casei = 1:4
    figure(casei)
    clf
    nv = [10 100 1000];
    MC = 1000;
    N = 50;
    Mp = 100;
    M = 100;

    lnpdf = @(y,mu,sigma2) -1/2*log(2*pi*sigma2) - 1/(2*sigma2)*(y-mu).^2;
    PFAs = zeros(MC,length(nv));
    for in = 1:3
        n = nv(in);
        for mc = 1:MC
            if casei == 1 % fix model
                ygiv = randn(n,1);
                sigma2s = ones([N,1]);
                mus = zeros([N,1]);
            elseif casei == 2 % fix but wrong model
                ygiv = rand(n,1);
                sigma2s = ones([N,1]);
                mus = zeros([N,1]);
            elseif casei == 3 % unknown parameters, correct class
                ygiv = randn(n,1);
                sigma2s = 1./gamrnd((n-3)/2,1/(.5*sum((ygiv-mean(ygiv)).^2)),[N,1]);
                mus = (randn([N,1]).*sqrt(sigma2s/n) + mean(ygiv));
            elseif casei == 4 % unknown parameters, wrong class
                ygiv = rand(n,1);
                sigma2s = 1./gamrnd((n-3)/2,1/(.5*sum((ygiv-mean(ygiv)).^2)),[N,1]);
                mus = (randn([N,1]).*sqrt(sigma2s/n) + mean(ygiv));
            end
            
            Tgiv = zeros(N,1);
            Tsim = zeros(N,M);
            PFA_o = zeros(N,1);
            for j = 1:N
                ysimp = randn(Mp,n).*sqrt(sigma2s(j)) + mus(j);
                zsimp = lnpdf(ysimp,mus(j),sigma2s(j));
                Ez = mean(zsimp(:));
                Vz = var(zsimp(:));
                ysim = randn(M,n).*sqrt(sigma2s(j)) + mus(j);
                zsim = lnpdf(ysim,mus(j),sigma2s(j));
                zgiv = lnpdf(ygiv,mus(j),sigma2s(j));
                Tgiv(j) = mean((zgiv-Ez).^2/Vz);
                Tsim(j,:) = mean((zsim-Ez).^2/Vz,2)';
                PFA_o(j) = mean(Tsim(j,:)<Tgiv(j));
            end
            PFAs(mc,in) = min(mean(PFA_o),1-mean(PFA_o));
            disp(num2str(mc))
        end
        subplot(3,1,in)
        histogram(PFAs(:,in),n,'BinLimits',[0 .5],'NumBins',20)
        
        title(['n = ',num2str(n)])
        drawnow
    end
end