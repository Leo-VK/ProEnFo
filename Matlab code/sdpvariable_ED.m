%% %define symbolic variable
% Unit output
P = sdpvar(Nunits,Horizon,'full');
% Transmission Lines fkiw f_l
flow = sdpvar(size(mpc.branch,1),Horizon,'full');
% %Transmission Network Nodes' Angle
angle = sdpvar(size(mpc.bus,1),Horizon,'full');
%Transmission line penalty
Fast_respond= sdpvar(3,Horizon,'full');
deep_peak= sdpvar(3,Horizon,'full');
energy_SD= sdpvar(3,Horizon,'full');
vSd = binvar(1,Horizon,'full');
vSc = binvar(1,Horizon,'full');