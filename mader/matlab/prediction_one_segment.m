
close all; clc;clear;
set(0,'DefaultFigureWindowStyle','docked') %'normal' 'docked'
set(0,'defaulttextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
%Let us change now the usual grey background of the matlab figures to white
set(0,'defaultfigurecolor',[1 1 1])

import casadi.*
addpath(genpath('./../../submodules/minvo/src/utils'));
addpath(genpath('./../../submodules/minvo/src/solutions'));
addpath(genpath('./more_utils'));

opti = casadi.Opti();
deg_pos_prediction=2;  
dim_pos=3;
% num_seg_prediction =1; %Not used (in this file we fit a polynomial, not a
% spline)
num_observations=10;
secs_prediction=100;

% This file follows the notation of https://otexts.com/fpp2/regression-matrices.html#regression-matrices
% Which is a generalization of the eq. 5.4 of https://otexts.com/fpp2/forecasting-regression.html 
% tmp=xnew*betaHat + 1.96*sigmaEhat*sqrt(1+xnew*inv(X'*X)*xnew')
% tmp=xnew*betaHat + 1.96*sigmaEhat*sqrt(1+(1/T) + ((tnew - mean(x))^2)/((T-1)*(std(x)^2)))
% The two formulas above give the same result for the case of linear regression

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%    PREDICTION     %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% all_t=linspace(t0,tf,10); 
all_t=MX.sym('all_t',1,num_observations);
all_pos=MX.sym('all_pos',dim_pos,num_observations);
% t_query=MX.sym('t',1,1);


all_t_modified=(all_t-max(all_t))/secs_prediction; %Shifting to avoid numerical issues

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Start of use of t_modified
X=[];
for i=0:deg_pos_prediction
    X=[X (all_t_modified').^i];
end

tmp=inv(X'*X)*X';
betaHat_x=tmp*all_pos(1,:)';
betaHat_y=tmp*all_pos(2,:)';
betaHat_z=tmp*all_pos(3,:)';

tmp=(1/(num_observations-deg_pos_prediction-1));
sigmaEhat_x=sqrt(tmp*norm(all_pos(1,:)'-X*betaHat_x)^2);
sigmaEhat_y=sqrt(tmp*norm(all_pos(2,:)'-X*betaHat_y)^2);
sigmaEhat_z=sqrt(tmp*norm(all_pos(3,:)'-X*betaHat_z)^2);


invXtX=inv(X'*X);
             
sigmaEhat=[sigmaEhat_x sigmaEhat_y sigmaEhat_z]';
coeff_mean=[flip(betaHat_x');
                 flip(betaHat_y');
                 flip(betaHat_z')];


T=[];
t=MX.sym('t',1,1);
for i=0:deg_pos_prediction
    T=[t^i; T];
end

tmp=(1+flip(T)'*invXtX*flip(T));
variance=(sigmaEhat.^2)*tmp; %this will be a polynomial of degree 2*deg_pos_prediction

coeff_variance=   [getCoeffPolyCasadi(variance(1), t, 2*deg_pos_prediction);
                   getCoeffPolyCasadi(variance(2), t, 2*deg_pos_prediction);
                   getCoeffPolyCasadi(variance(3), t, 2*deg_pos_prediction)];
    

%Both coeff_predicted and coeff_variance are expressed such that t=0
%corresponds with max(all_t);
g= Function('g', {all_t, all_pos},      [{coeff_mean, coeff_variance, secs_prediction}], ...
                 {'all_t','all_pos'},    {'coeff_mean', 'coeff_variance', 'secs_prediction'} );


all_pos_value= [linspace(0.0,10,num_observations);
                linspace(0.0,10,num_observations);
                linspace(0.0,10,num_observations)] + 4*rand(dim_pos,num_observations);
            
t0=1e6 + 5;
tf=1e6 + 10.5;
all_t_value=linspace(t0,tf,num_observations); 
tic
sol=g('all_t', all_t_value, 'all_pos', all_pos_value );
toc
coeff_mean_value=full(sol.coeff_mean);

syms t real
T=[];
for i=0:deg_pos_prediction
    T=[t^i; T];
end

T2d=[];
for i=0:2*deg_pos_prediction
    T2d=[t^i; T2d];
end

coeff_variance=full(sol.coeff_variance);

mean=coeff_mean_value*T;

c_prediction_value=1.96; %For x% bands, set this to norminv(x) (97.5% band <--> 1.96, see https://en.wikipedia.org/wiki/1.96  ). See also norminv(0.975,mu, sigma)
bands_up=mean   + c_prediction_value*sqrt(coeff_variance*T2d);
bands_down=mean - c_prediction_value*sqrt(coeff_variance*T2d);

all_t_modified_value=(all_t_value-max(all_t_value))/secs_prediction; %Shifting because both the means and the bands are expressed wrt the 

figure; hold on;
% interv=[min(all_t_modified_value), max(all_t_modified_value)+secs_pred];
interv=[min(all_t_modified_value),0.1];
for i=1:3
    subplot(3,1,i); hold on;
    plot(all_t_modified_value, all_pos_value(i,:), 'o')
    fplot(mean(i), interv)
    fplot(bands_up(i), interv, '--')
    fplot(bands_down(i), interv, '--')
end

% subplot(3,1,2)
% plot(all_t_modified_value, all_pos_value(1,:), 'o')
% fplot(coeff_predicted_value(1,:)*T, interv)
% fplot(bands_up(1), interv, '--')
% fplot(bands_down(1), interv, '--')

% betaHat_xcoeff_predicted_value*T
% t_original=MX.sym('t_original',1,1);
% t=MX.sym('t',1,1);
% T=[];
% for i=0:deg_pos_prediction
%     T=[t^i; T];
% end
% coeff_x=getCoeffPolyCasadi(substitute(coeff_predicted(1,:)*T, t, t-offset), t, deg_pos_prediction);
% coeff_y=getCoeffPolyCasadi(substitute(coeff_predicted(2,:)*T, t, t-offset), t, deg_pos_prediction);
% coeff_z=getCoeffPolyCasadi(substitute(coeff_predicted(3,:)*T, t, t-offset), t, deg_pos_prediction);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of use of t_modified

% coeff_predicted=[coeff_x;
%                  coeff_y;
%                  coeff_z];

%%

% clc;
% rng('default') % For reproducibility
% x = -5:5;
% 
% T=numel(x);  %T is the number of observations
% k=2;
% 
% y = - 20*x - 3 + 5*randn(size(x));
% 
% 
% X=[x'.^0    x'.^1  ];
% 
% 
% % 
% % figure;
% % degree = 2; % Degree of the fit
% % [p,S] = polyfit(x,y',degree);
% % 
% % alpha = 0.05; % Significance level
% % [yfit,delta] = polyconf(p,x,S,'alpha',alpha);
% % 
% % 
% % 
% % plot(x,y,'b+')
% % hold on
% % plot(x,yfit,'g-')
% % plot(x,yfit-delta,'r--',x,yfit+delta,'r--')
% % % legend('Data','Fit','95percent Prediction Intervals')
% % title(['Fit: ']) %,texlabel(polystr(round(p,2)))])
% 
% 
% y=y';
% betaHat=inv(X'*X) *X' * y;
% sigmaEhat=sqrt((1/(T-k-1))*(y-X*betaHat)'*(y-X*betaHat)); %k is the degree of the polynomial fitted, T is the number of observations
% 
% syms tnew real
% tnew=5
% xnew=[tnew'.^0    tnew'.^1  ];
% 
% %Last equation of https://otexts.com/fpp2/regression-matrices.html#regression-matrices
% tmp=xnew*betaHat + 1.96*sigmaEhat*sqrt(1+xnew*inv(X'*X)*xnew')
% 
% %Eq 5.4 of https://otexts.com/fpp2/forecasting-regression.html
% tmp=xnew*betaHat + 1.96*sigmaEhat*sqrt(1+(1/T) + ((tnew - mean(x))^2)/((T-1)*(std(x)^2)))
% 
% %Note that the two equations above give the same result (the second one is
% %only valid for regression b0 + b1x, while the first one is more general
% 
% % fplot(tmp, [0,5])
% % p*[tnew'.^0    tnew'.^1   tnew'.^2]'
% 
% 
% 


