clear all;
close all;

mpc = loadcase(case30);
Parameter_ED;
loa=csvread('E:/ETH learning material/Master Thesis/Code/train_test_d_3_h_4/y_train_zone_1_station_2_REC_norm_minmax.csv');
saple=loa(54*24+1:55*24);
saple_max=max(saple);
saple_min=min(saple);
gentotal_max=sum(mpc.gen(:,9));
gentotal_min=sum(mpc.gen(:,10));
exp_load1=0.8*((gentotal_max-gentotal_min)/(saple_max-saple_min).*(saple-saple_min)+gentotal_min)+20;
exp_load=6*(exp_load1+addin+extra_addin);

sdpvariable_ED;
Actual_Load = mpc.bus(:,3)/sum(mpc.bus(:,3)).*repmat(exp_load',size(mpc.bus,1),1);
Constraintsetting_ED;
largest_up=max(1.1*exp_load(2:end)-0.9*exp_load(1:end-1));
largest_down=min(0.9*exp_load(2:end)-1.1*exp_load(1:end-1));

NodeBalance =       Unit2Node'*P == Actual_Load+Line2Node'*flow;
PowerFlows =        flow     ==  mpc.baseMVA*repmat((1./mpc.branch(:,4))',Horizon,1)'.*(Line2Node*angle);
LineLimits =       -500*ones(size(mpc.branch,1),Horizon)<=flow<=500*ones(size(mpc.branch,1),Horizon);
Angle =            -pi    <= angle <= pi;
in=find(mpc.bus(:,2)==3);
AngleSlackBus =     angle(in,:)  == zeros(1,Horizon);
Idealgrid = [NodeBalance,PowerFlows,LineLimits, Angle,AngleSlackBus];
Objective=sum(Q*(P.*P)+Marginal_C*P);
Constraints_ideal =[Idealgrid,Minmaxpower,Rampuplimit,Rampdownlimit];
sol_ideal=optimize(Constraints_ideal,Objective,ops);
ideal_totalcost =value(Objective);
idealP=value(P);
idealFlow=value(flow);
idealAngle=value(angle);
Margin_IDEAL=Q'.*idealP+Marginal_C';
Totcost_IDEAL=Q'.*(idealP.*idealP)+Marginal_C'.*idealP;
ideal_sumload=sum(Actual_Load,1);
test_sumgene=sum(idealP,1);
branch_flowmargin=1000*ones(size(mpc.branch,1),1)-max(abs(idealFlow),[],2);

end
