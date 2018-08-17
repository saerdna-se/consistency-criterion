function [z] = z_kangaroo(N,y,theta,transition,observation_pdf_log)
%Z_KANGAROO Particle filter estimating z_i for the kangaroo example
%   Source code for the kangaroo count example in
%   "Data Consistency Approach to Model Validation",
%   Andreas Svensson, Dave Zachariah, Petre Stoica, Thomas B. Schön
%
%   Code Andreas Svensson 2018

sigma = theta(1);
tau = theta(2);

% Particle filter
T = length(y);
x_pf = zeros(N,T);
w_log = zeros(N,T);

x_pf(:,1) = exp(mvnrnd(0*ones(N,1),5));
w_log(:,1) = observation_pdf_log(y(1,1),x_pf(:,1),tau)+observation_pdf_log(y(2,1),x_pf(:,1),tau);
w = exp(w_log-max(w_log));

for t = 2:T
    a = systematic_resampling(w,N);
    x_pf(:,t) = transition(x_pf(a,t-1),sigma,t);
    w_log(:,t) = observation_pdf_log(y(1,t),x_pf(:,t),tau)+observation_pdf_log(y(2,t),x_pf(:,t),tau);
    w = exp(w_log(:,t)-max(w_log(:,t)));
end

z = log(mean(exp(w_log)));

end

