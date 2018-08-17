% Source code for the earthquake count example in
% "Data Consistency Approach to Model Validation",
% Andreas Svensson, Dave Zachariah, Petre Stoica, Thomas B. Schön
%
% Code Andreas Svensson 2018

clear

dataset = 8;

if dataset==5
    load('earthquake_data/e5.mat')
    ygiv = e5';
elseif dataset==6
    load('earthquake_data/e6.mat')
    ygiv = e6';
elseif dataset==7
    load('earthquake_data/e7.mat')
    ygiv = e7';
elseif dataset==8
    load('earthquake_data/e8.mat')
    ygiv = e8';
end

n = length(ygiv);

N = 500;
M = 500;


%% Poisson model
lambdas = gamrnd(sum(ygiv)+1,1/n,[N 1]);
pfau_poisson = zeros(N,1);

for Ni = 1:N
    ysim = poissrnd(lambdas(Ni),[M n]);
    zsim = log(poisspdf(ysim,lambdas(Ni)));
    zgiv = log(poisspdf(ygiv,lambdas(Ni)));
    Ez = mean(zsim(:));
    Vz = var(zsim(:));
    Tgiva = mean((zgiv-Ez).^2/Vz);
    Tsima = mean((zsim-Ez).^2/Vz,2);
    pfau_poisson(Ni) = mean(Tsima<Tgiva);
end


disp(['Poisson: PFA = ',num2str(min(1-mean(pfau_poisson),mean(pfau_poisson)))])


%% negative binomial model

% sample from parameter posterior
RP = nbinposterior(ygiv,N);

pfau_nb = zeros(N,1);

for Ni = 1:N
    ysim = nbinrnd(RP(Ni,1),RP(Ni,2),[M n]);
    zsim = log(nbinpdf(ysim,RP(Ni,1),RP(Ni,2)));
    zgiv = log(nbinpdf(ygiv,RP(Ni,1),RP(Ni,2)));
    Ez = mean(zsim(:));
    Vz = var(zsim(:));
    Tgiva = mean((zgiv-Ez).^2/Vz);
    Tsima = mean((zsim-Ez).^2/Vz,2);
    pfau_nb(Ni) = mean(Tsima<Tgiva);
end

disp(['Neg bin: PFA = ',num2str(min(1-mean(pfau_nb),mean(pfau_nb)))])