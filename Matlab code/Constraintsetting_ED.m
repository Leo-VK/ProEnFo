%% Ramping up/down constraints (reserves are regarded seperate units)
% Rampuplimit =  P(:,1)-Pini <= Ramup';
Rampuplimit = [P(:,1)-Pini <= Ramup', P(:,2:Horizon)-P(:,1:Horizon-1) <= repmat(Ramup,Horizon-1,1)'];

% Rampdownlimit = Pini-P(:,1) <= Ramdown';
Rampdownlimit = [Pini-P(:,1) <= Ramdown', P(:,1:Horizon-1)-P(:,2:Horizon) <= repmat(Ramdown,Horizon-1,1)'];
%% Min max power contraints
Minmaxpower = repmat(Pmin,1,Horizon) <= P <= repmat(Pmax,1,Horizon);
%% For yalmip 5
% Minpower_actual = repmat(Pmin,1,Horizon)+ res_up <= P;
% Maxpower_actual = P+res_down <= repmat(Pmax,1,Horizon);
% Rampuplimit_actual =  P(:,1)+res_up(:,1)-Pini <= Ramup';
% Rampuplimit_actual = [Rampuplimit_actual, P(:,2:Horizon)+res_up(:,2:Horizon)-P(:,1:Horizon-1) <= repmat(Ramup,Horizon-1,1)'];
% 
% Rampdownlimit_actual = Pini-P(:,1)+res_down(:,1)<= Ramdown';
% Rampdownlimit_actual = [Rampdownlimit_actual, P(:,1:Horizon-1)-P(:,2:Horizon)+res_down(:,2:Horizon) <= repmat(Ramdown,Horizon-1,1)'];
