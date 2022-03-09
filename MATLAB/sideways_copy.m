syms s t x p1  p2 p3;

DiffDriveGantry;
% V_t = 24.0*10/128*heaviside(t)-24.0*10/128*heaviside(t-0.5);  % step
V_t = 24.0*10/128*heaviside(t)-24.0*20/128*heaviside(t-0.5)+24.0*10/128*heaviside(t-1);  % double step
% V_t = 24*21/128*heaviside(t)*t - 24*21/128*t*heaviside(t-0.5);  % ramp
Hss = p1*s^3+p2*s^2+p3*s;
invHt = ilaplace(inv(Hss),t);
invHt_exp = 1/p3*(1-((exp(t*sqrt((p2/(2*p1))^2-p3/p1)-t*p2/(2*p1))+exp(-t*sqrt((p2/(2*p1))^2-p3/p1)-t*p2/(2*p1)))/2+(exp(t*sqrt((p2/(2*p1))^2-p3/p1)-t*p2/(2*p1))-exp(-t*sqrt((p2/(2*p1))^2-p3/p1)-t*p2/(2*p1)))/(2*sqrt((p2/(2*p1))^2-p3/p1))));
invHt_func = matlabFunction(invHt_exp);
theta_symb = int(V_t*invHt_exp,t,0,x);


p1_start = L/K*(J_L+J_S);
p2_start = (R*(J_L+J_S)+L*(bl+bs))/K;
p3_start = K + R*(b1+bs)/K;

theta_func = matlabFunction(theta_symb);
% Table = readtable('New_PS_ID/x/step_x_speed_10.txt','NumHeaderLines',1);
Table = readtable('New_PS_ID/x/d_step_x_speed_10.txt','NumHeaderLines',1);
% Table = readtable('New_PS_ID/x/ramp_x_uptoSpeed_21.txt','NumHeaderLines',1);
t = table2array(Table(:,6));
t = t - t(1);
theta_exp = table2array(Table(:,1))*pi/180;
theta_exp = theta_exp - theta_exp(1);
[fitted_curve,gof] = fit(t,theta_exp,theta_func,'StartPoint', ...
    [p1_start,p2_start,p3_start])
p = coeffvalues(fitted_curve);
theta_pred = theta_func(p(1),p(2),p(3),t);
hold on
plot(t,theta_pred);
plot(t,theta_exp);
legend("Predicted","Experimental","Location", "best")
hold off
