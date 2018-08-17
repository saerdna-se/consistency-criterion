% Source code for the polynomial regression example in
% "Data Consistency Approach to Model Validation",
% Andreas Svensson, Dave Zachariah, Petre Stoica, Thomas B. Schön
%
% Code Andreas Svensson 2018

clear, clf
figure(1)
rng(1)

n = 50;
sigma = 2;
x = linspace(-25,25,n);

lnp = @(yi,sigma2e,mue) -.5*log(2*pi*sigma2e)-.5/sigma2e*(mue-yi).^2;

MC = 1;
poly3 = 4*x+.5*x.^2-.2*x.^3;
N = 100;
M = 100;
pfas = zeros(1,MC);

% data from 3rd order polynomial + noise, model class is polynomial of various orders + noise
ygiv = poly3 + randn(1,n)*sigma;

for order = 1:3
    if order == 1
        phi = [ones(n,1) x'];
        dp = order + 1;
    elseif order == 2
        phi = [ones(n,1) x' x'.^2];
        dp = order + 1;
    elseif order == 3
        phi = [ones(n,1) x' x'.^2 x'.^3];
        dp = order + 1;
    end

    % sample models of given order
    xxt = zeros(dp,dp);
    yxt = zeros(1,dp);
    for i = 1:n
        xxt = xxt + phi(i,:)'*phi(i,:);
        yxt = yxt + ygiv(i)'*phi(i,:);
    end
    sigma2s = 1./gamrnd(((n)-3)/2,2/(sum(ygiv.^2) - yxt*(xxt\yxt')),[N,1]);
    post_mean = yxt/xxt;
    coeffs = zeros(dp,N);
    for j = 1:N
        coeffs(:,j) = sigma2s(j)*randn([1 dp])/(xxt) + post_mean;
    end
    
    pfau = zeros(1,N);
    for j = 1:N
    
        if order == 1
            mue = coeffs(1,j) + x*coeffs(2,j);
        elseif order == 2
            mue = coeffs(1,j) + x*coeffs(2,j) + x.^2*coeffs(3,j);
        elseif order == 3
            mue = coeffs(1,j) + x*coeffs(2,j) + x.^2*coeffs(3,j) + x.^3*coeffs(4,j);
        end
        
        zgiv = lnp(ygiv,sigma2s(j),mue);
        
        ysim = zeros(M,n);
        zsim = zeros(M,n);
        for k = 1:M
            ysim(k,:) = mue + randn(1,n)*sqrt(sigma2s(j));
            zsim(k,:) = lnp(ysim(k,:),sigma2s(j),mue);
        end
        Ez = mean(zsim(:));
        Vz = var(zsim(:));

        Tgiva = mean((zgiv-Ez).^2/Vz);
        Tsima = mean((zsim-Ez).^2/Vz,2);
        pfau(j) = mean(Tsima>Tgiva);
    end
    pfas = min(mean(pfau),1-mean(pfau));

    % plot data and mean model
    if order == 1
        mueav = post_mean(1) + x*post_mean(2);
    elseif order == 2
        mueav = post_mean(1) + x*post_mean(2) + x.^2*post_mean(3);
    elseif order == 3
        mueav = post_mean(1) + x*post_mean(2) + x.^2*post_mean(3) + x.^3*post_mean(4);
    end
        
    subplot(3,1,order)
    plot(x,ygiv,'b.','markersize',20)
    hold on
    plot(x,mueav,'k','linewidth',2)
    plot(x,mueav+2*sqrt(mean(sigma2s)),'--k')
    plot(x,mueav-2*sqrt(mean(sigma2s)),'--k')
    legend('Data', ['Estimated ',num2str(order),' order model mean'], ['Estimated ',num2str(order),' order model variance'])
    set(gca,'xtick',-20:10:20)
    set(gca,'ytick',-2000:2000:3000)
    title(['PFA = ',num2str(pfas)])
    drawnow
end
