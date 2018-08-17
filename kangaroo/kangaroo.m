% Source code for the kangaroo count example in
% "Data Consistency Approach to Model Validation",
% Andreas Svensson, Dave Zachariah, Petre Stoica, Thomas B. Schön
%
% Code Andreas Svensson 2018
clear
rng(1)

% data
kangaroo_data = csvread('kangaroo.csv',1,1);
T = length(kangaroo_data);
T_vec = kangaroo_data(3,:);
y_giv = kangaroo_data(1:2,:);

% model
transition = @(x,sigma,t) x.*exp(sigma*sqrt(T_vec(t)-T_vec(t-1))*randn(size(x)));
observation_rnd = @(x,tau) nbinrnd(1/tau, tau*x./(x*tau+1));
observation_pdf_log = @(y,x,tau) nbinpdf_log(y,1/tau, tau*x./(x*tau+1));

% infer parameter posterior with Metropolis-Hastings
N = 1000;
thetas = zeros(2,N);
ll = zeros(1,N);
thetas(:,1) = [.5;.1];
ll(1) = sum(z_kangaroo(10000,y_giv,thetas(:,1),transition,observation_pdf_log));
for k = 2:N
    theta_p = [0;0];
        while((theta_p(1)<=0)||(theta_p(1)>=10)||(theta_p(2)<=0)||(theta_p(2)>=10))
            theta_p = thetas(:,k-1) + randn([2 1])*.05;
        end
%     end
    ll_p = sum(z_kangaroo(10000,y_giv,theta_p,transition,observation_pdf_log));
    d = rand;
    if exp(ll_p-ll(k-1))>d
        % accept
        thetas(:,k) = theta_p;
        ll(k) = ll_p;
        disp(['k = ',num2str(k),'. accept!'])
    else
        % reject
        thetas(:,k) = thetas(:,k-1);
        ll(k) = ll(k-1);
        disp(['k = ',num2str(k),'. reject'])
    end
    if round(k/100)*100==k
        clf
        plot(thetas');
        drawnow
    end
end

M = 200;
Mp = 200;

pfau = zeros(1,N);
x_sim = zeros(1,T);
y_sim = zeros(2,T);
D = zeros(1,M);

z_simp = zeros(Mp,T);
z_sim = zeros(M,T);
T_sim = zeros(1,M);

for i = 1:N
    % Simulate Mp trajectories for estimating Ez and Varz
    for j = 1:Mp
            x_sim(1) = exp(mvnrnd(0,5));
            y_sim(1,1) = observation_rnd(x_sim(1),thetas(2,i));
            y_sim(2,1) = observation_rnd(x_sim(1),thetas(2,i));
            
            for t = 2:T
                x_sim(t) = transition(x_sim(t-1),thetas(1,i),t);
                y_sim(1,t) = observation_rnd(x_sim(t),thetas(2,i));
                y_sim(2,t) = observation_rnd(x_sim(t),thetas(2,i));
            end

        % Compute z for each trajectory
        z_simp(j,:) = z_kangaroo(1000,y_sim,thetas(:,i),transition,observation_pdf_log);
    end
    
    Ez = mean(z_simp);
    Vz = var(z_simp);
    
    % Simulate M trajectories
    for j = 1:M
            x_sim(1) = exp(mvnrnd(0,5));
            y_sim(1,1) = observation_rnd(x_sim(1),thetas(2,i));
            y_sim(2,1) = observation_rnd(x_sim(1),thetas(2,i));
            
            for t = 2:T
                x_sim(t) = transition(x_sim(t-1),thetas(1,i),t);
                y_sim(1,t) = observation_rnd(x_sim(t),thetas(2,i));
                y_sim(2,t) = observation_rnd(x_sim(t),thetas(2,i));
            end

        % Compute T for each trajectory
        z_sim(j,:) = z_kangaroo(1000,y_sim,thetas(:,i),transition,observation_pdf_log);
        T_sim(j) = mean((z_sim(j,:)-Ez).^2./Vz);
    end
    
    % Compute T for given data with parameter i
    z_giv = z_kangaroo(1000,y_giv,thetas(:,i),transition,observation_pdf_log);
    
    % 
    T_giv = mean((z_giv-Ez).^2./Vz);
    
    % Set rho(i)
    pfau(i) = mean(T_sim<T_giv);
    disp(['i = ',num2str(i),'/',num2str(N),'. PFAU so far: ',num2str(mean(pfau(1:i)))])
end
pfas = min(mean(pfau),1-mean(pfau));

disp(pfas);
