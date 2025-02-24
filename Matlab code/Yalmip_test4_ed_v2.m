%% -------------------------Final version of Loss function generator (piecewise)-----------------------------%

clear all;
% clc;
close all;
%% Grid parameter setting
mpc = loadcase(case30);
Parameter_ED;
% loa=csvread('C:/Users/jialzhan/Desktop/Master Thesis/train_test_d_3_h_4/y_train_zone_1_station_2_REC_norm_minmax.csv');
loa=csvread('E:/ETH learning material/Master Thesis/Code/train_test_d_3_h_4/y_train_zone_1_station_2_REC_norm_minmax.csv');
% plot(loa')
saple=loa(54*24+1:55*24);%for example take the samples from the actual loading at day 4 to 5
saple_max=max(saple);
saple_min=min(saple);
gentotal_max=sum(mpc.gen(:,9));
gentotal_min=sum(mpc.gen(:,10));
exp_load1=0.8*((gentotal_max-gentotal_min)/(saple_max-saple_min).*(saple-saple_min)+gentotal_min)+20;
addin=zeros(Horizon,1);
addin(1:5)=80;
addin(6)=50;
addin(7)=-50;
addin(8)=-40;
addin(9)=20;
addin(10)=55;
addin(11)=50;
addin(12)=60;
addin(13:16)=60;
addin(17)=60;
addin(18)=30;
addin(19)=-30;
addin(20)=-25;
addin(21:22)=-5;
addin(23)=40;
addin(24)=60;
extra_addin=40*ones(24,1);
exp_load=6*(exp_load1+addin+extra_addin);
% plot(exp_load')
% set(gca,'XTick',(1:24:60*24))
%% Decision variable declaration
sdpvariable_ED;
%% Set up forecast load and system constraints
% fprintf('Miu equals to: %f \n',miu);
Actual_Load = mpc.bus(:,3)/sum(mpc.bus(:,3)).*repmat(exp_load',size(mpc.bus,1),1);%repmat(mpc.bus(:,3),1,Horizon) + repmat(2*sin((1:Horizon)*2*pi/24)+10*(1:Horizon)*2*pi/24/7,size(mpc.bus,1),1);
% figure()
% plot(exp_load')
% exp_load_forecast=exp_load'.*stac_sce;
% figure()
% plot(1:24, exp_load_forecast')
% xlim([1 24])
% xlabel('Hour');
% ylabel('Total Load (MW)');
% figs = findobj(0, 'type', 'figure');
% % print each figure in figs to a separate .eps file
% print(figs(1), '-depsc', sprintf('Monte.eps'))
Constraintsetting_ED;
largest_up=max(1.1*exp_load(2:end)-0.9*exp_load(1:end-1));
largest_down=min(0.9*exp_load(2:end)-1.1*exp_load(1:end-1));
%% -------------------------Simulate Ideal case operation-----------------------------%
%% Actual grid constraint
% RATE_A column in branch matrix represent the flow limits ->P61, opf.flow_lim define the type of flow constraint
% mpc.bus(:,3) indicates those bus with actual load connected to (P_demand is not zero)
% equation (1b) power balance at all buses
% Actual_Load=repmat(mpc.bus(:,3),1,Horizon)+(-5+(10+5).*rand(size(mpc.bus,1),Horizon));
% plot(Actual_Load')
NodeBalance =       Unit2Node'*P == Actual_Load+Line2Node'*flow;
% equation (1m) power flow calculation using DC power flow
PowerFlows =        flow     ==  mpc.baseMVA*repmat((1./mpc.branch(:,4))',Horizon,1)'.*(Line2Node*angle);
% equation (1n)
LineLimits =       -500*ones(size(mpc.branch,1),Horizon)<=flow<=500*ones(size(mpc.branch,1),Horizon);%repmat(1.1.*mpc.branch(:,6),1,Horizon);%100*ones(size(mpc.branch,1),Horizon)
Angle =            -pi    <= angle <= pi;
% Transpenalty =             e_tran>=0;
%Slack Bus
in=find(mpc.bus(:,2)==3);
AngleSlackBus =     angle(in,:)  == zeros(1,Horizon);
Idealgrid = [NodeBalance,PowerFlows,LineLimits, Angle,AngleSlackBus];%Transpenalty,
%% Set up objective function
Objective=sum(Q*(P.*P)+Marginal_C*P);% P(:,k)'*Q*P(:,k)+sum(sum(4e3.*e_tran))+
%% Solving Ideal case problem
ops = sdpsettings('solver','gurobi','verbose',1,'debug',1);%
Constraints_ideal =[Idealgrid,Minmaxpower,Rampuplimit,Rampdownlimit];%,actualbalance,
sol_ideal=optimize(Constraints_ideal,Objective,ops);
if sol_ideal.problem~=0
sol_ideal.info
yalmiperror(sol_ideal.problem)
end
ideal_totalcost =value(Objective);
idealP=value(P);
idealFlow=value(flow);
idealAngle=value(angle);
Margin_IDEAL=Q'.*idealP+Marginal_C';
Totcost_IDEAL=Q'.*(idealP.*idealP)+Marginal_C'.*idealP;
ideal_sumload=sum(Actual_Load,1);
test_sumgene=sum(idealP,1);
branch_flowmargin=1000*ones(size(mpc.branch,1),1)-max(abs(idealFlow),[],2);
if all(abs(ideal_sumload-test_sumgene)>=1e-2)
    warning('Ideal case generations failed to satisfy demand')
end
% Record all the cost for each time unit of ideal case
ideal_segment_cost=[];

for t=1:Horizon
    ideal_segment_cost=[ideal_segment_cost,Q*(idealP(:,t).*idealP(:,t))+Marginal_C*idealP(:,t)];%+
end
% t=9;
% Marginal_C*idealP(:,t)+Cnoload*idealstate(:,t)+CSU*ideal_y(:,t)+CSD*ideal_z(:,t)
%% -------------------------Simulate Day ahead UC operation-----------------------------%
% Generate N prediction scenarios
N=10000;
% discrete_MPE=0.9:0.01:1.1;
stac_sce=[];
for scen=1:Horizon
    rng(scen);
%     r=datasample(discrete_MPE,N)';
%     r=normrnd(1,0.033,[N,1]);
    r=unifrnd(0.9,1.1,[N,1]);
    stac_sce=[stac_sce,r];
end
for k=1:N
Forecast_Load(:,:,k)=Actual_Load.*stac_sce(k,:);
end
% histogram(stac_sce(:,24))
% c = arrayfun(@(x)length(find(stac_sce(:,10) == x)), unique(stac_sce(:,10)), 'Uniform', false);
% cell2mat(c)
% dfdfd=Forecast_Load(:,:,1);
% plot(Forecast_Load(:,:,6)')
% cases=19; 
PCE=[];
PCE_SAME=[];
MPE=[];
MPE_ele=[];
PCE_DAUC=[];
PCE_DAUC2=[];
PCE_ACT=[];
PCE_ACT2=[];
PCE_ADD=[];
PCE_ADD2=[];
Actual_new=[];
Actual_fa=[];
Actual_pe=[];
% DAUC_Flow_flowmargin=[];
%% Predict grid constraint
tic
for cases=1:N%275:275,345:345%293/406/316/266
fprintf('This is the case: %d \n',cases);
% equation (1b) power balance at all buses
NodeBalance_forecast =       Unit2Node'*P == Forecast_Load(:,:,cases)+Line2Node'*flow;
Realgrid = [NodeBalance_forecast,PowerFlows,LineLimits, Angle,AngleSlackBus];

% predictbalance=[];
% for k = 1:Horizon
%     predictbalance = [predictbalance,sum(P(:,k)) == sum(Forecast_Load(:,k,cases))];
% end
%% Determine DAUC based on forecast load
Constraints_DAUC =[Realgrid,Minmaxpower,Rampuplimit,Rampdownlimit];%predictbalance
sol_DAUC=optimize(Constraints_DAUC,Objective,ops);
if sol_DAUC.problem~=0
sol_DAUC.info
yalmiperror(sol_DAUC.problem)
end
DAUC_totalcost = value(Objective);
DAUC_P = value(P);
DAUC_Flow=value(flow);

DAUC_Angle=value(angle);
% fprintf('DAUC Total cost is:%d \n',DAUC_totalcost);
DAUC_sumload=sum(Forecast_Load(:,:,cases),1);
DAUC_sumgene=sum(value(DAUC_P),1);
DAUC_pchange=DAUC_P(:,2:end)-DAUC_P(:,1:end-1);
Margin_DAUC=Q'.*DAUC_P+Marginal_C';
Totcost_DAUC=Q'.*(DAUC_P.*DAUC_P)+Marginal_C'.*DAUC_P;
% DAUC_Flow_flowmargin=[DAUC_Flow_flowmargin,max(abs(DAUC_Flow),[],2)-mpc.branch(:,6)];
if all(abs(DAUC_sumload-DAUC_sumgene)>=1e-2)
    warning('DAUC schedule failed to satisfy demand')
end
% DAUC_segment_cost=[];
% for t=1:Horizon
%     DAUC_segment_cost=[DAUC_segment_cost,Q*(DAUC_P(:,t).*DAUC_P(:,t))+Marginal_C*DAUC_P(:,t)];%+
% end

%% -------------------------Simulate Real-time operation-----------------------------%
% Add BESS capacity constraints, not mentioned in the paper
reserve_up = 0 <= Fast_respond <= repmat([20;40;120],1,Horizon).*repmat(vSd,3,1);
reserve_down = 0 <= deep_peak <= repmat([20;40;120],1,Horizon).*repmat(vSc,3,1);
SDnoCD =                   vSc + vSd <= 1;

% %%Energy limits, Energy balance equation

% Emax = [500;500;500];
Eini = [800;800;800];
SDEnergyLimitdn =  energy_SD >=0;% <= repmat(Emax,1,Horizon);

SDEnergyBalance =          energy_SD(:,1) == Eini - Fast_respond(:,1) + deep_peak(:,1);
SDEnergyBalance =          [SDEnergyBalance, energy_SD(:,2:Horizon) == energy_SD(:,1:Horizon-1) - Fast_respond(:,2:Horizon) + deep_peak(:,2:Horizon)];   
% SDEnergyBalance =          [SDEnergyBalance energy_SD(:,Horizon) == Eini];

%Reserve limits (Upward/Downward reserve capacity) 
% RU = repmat([1;1;1;10;10;5],1,Horizon);%[1;1;1;10;10;5]
% RD = repmat([5;10;10;1;1;1],1,Horizon);%[15;15;20;5;10;10]
% RD = repmat([5;5;5;10;10;10],1,Horizon);
% RU = repmat([10;10;10;5;5;5],1,Horizon);
RD = repmat(Pmax.*0.012,1,Horizon);
RU = RD;
intermpc=mpc.bus(mpc.bus(:,2)==1,:);
[ivalue,idyy]=max(intermpc(:,3));
[ivalue2,idyy2] = max(intermpc(intermpc(:,3)<ivalue,3));
[ivalue3,idyy3] = max(intermpc(intermpc(:,3)<ivalue2,3));
a_1=intermpc(idyy,1);
a_2=intermpc(find(intermpc(:,3)==ivalue2),1);
a_3=intermpc(find(intermpc(:,3)==ivalue3),1);
SD2Node=zeros(3,size(mpc.bus,1));
SD2Node(1,a_1)=1;
SD2Node(2,a_2)=1;
SD2Node(3,a_3)=1;

%Update NodeBalance Constraint adding the storage device charge/discharge
actual_balance = Unit2Node'*P + SD2Node'*Fast_respond == Actual_Load+Line2Node'*flow+SD2Node'*deep_peak;

% actual_balance=[];%
% % 
% for k = 1:Horizon
%     if stac_sce(cases,k)>1
%         actual_balance = [actual_balance,Unit2Node'*P(:,k)-SD2Node'*deep_peak(:,k) == Actual_Load(:,k)+Line2Node'*flow(:,k),Fast_respond(:,k)==0];%,P(:,k)<=DAUC_P(:,k) (Unit2Node'*(P(:,k)-RD(:,k)) >= Actual_Load(:,k)+Line2Node'*flow(:,k)),RU(:,k)==zeros(Nunits,1)
%     elseif stac_sce(cases,k)<1
%         actual_balance = [actual_balance,Unit2Node'*P(:,k)+SD2Node'*Fast_respond(:,k)== Actual_Load(:,k)+Line2Node'*flow(:,k),deep_peak(:,k)==0]; % ,P(:,k)>=DAUC_P(:,k)(Unit2Node'*(P(:,k)+RU(:,k)) >= Actual_Load(:,k)+Line2Node'*flow(:,k)-epsillon(:,k)),RD(:,k)==zeros(Nunits,1)
%     else
%         actual_balance = [actual_balance,Unit2Node'*P(:,k) == Actual_Load(:,k)+Line2Node'*flow(:,k),Fast_respond(:,k)==0,deep_peak(:,k)==0];
%     end
% end

% Day ahead units' state cannot be varied
% Add reserve constraint (6)
Minmaxpower_reserve = (DAUC_P-RD) <= P <= (DAUC_P+RU);%.*DAUC_state
BESS = [actual_balance,SDEnergyLimitdn,SDEnergyBalance];%
Actualgrid = [BESS,PowerFlows,LineLimits, Angle,AngleSlackBus];%actual_balance,
% real_constraint=[equality,Minmaxpower_reserve,actual_balance];%,actualbalance_1
%Add load shedding penalty
% Deficit_response= Fast_respond >= 0;
% Surplus_response= deep_peak >= 0;
% fast_price= [50,48,52];%*0.2*0.05*0.95*1e4;%0.01*[100,50,150];%1.5*(repmat(Q',1,Horizon).*DAUC_P+repmat(Marginal_C',1,Horizon).*DAUC_P);%%[250,150,50,350,240,200];
% shed_price= [10,10,1];%*0%0.01*fast_price;%60,40,80//70,50,100
% fast_price= [45,60,120];%*0.2*0.05*0.95*1e4;%0.01*[100,50,150];%1.5*(repmat(Q',1,Horizon).*DAUC_P+repmat(Marginal_C',1,Horizon).*DAUC_P);%%[250,150,50,350,240,200];

fast_price_1=repmat([25;45;60],1,4);%[15;35;50],1,4
fast_price_ed=repmat([20;50;60],1,2);%5-6
fast_price_2=repmat([25;50;65],1,7);%7-13
fast_price_1415=repmat([25;50;60],1,2);%14-15
fast_price_3=repmat([30;55;70],1,3);%16-18
fast_price_19=repmat([40;65;90],1,1);%19
fast_price_20=repmat([50;90;100],1,1);%20
fast_price_4=repmat([50;70;90],1,3);%21-23
fast_price_5=repmat([25;45;70],1,1);%24
fast_price=[fast_price_1,fast_price_ed,fast_price_2,fast_price_1415,fast_price_3,fast_price_19,fast_price_20,fast_price_4,fast_price_5];

shed_price_1=repmat([0;2;20],1,1);%1
shed_price_ed=repmat([0;8;20],1,3);%2-4
shed_price_2=repmat([0;0;15],1,15);%5-19
shed_price_3=repmat([0;2;15],1,2);%20-21
shed_price_4=repmat([0;1;20],1,3);%22-24
shed_price=[shed_price_1,shed_price_ed,shed_price_2,shed_price_3,shed_price_4];


penalty=[reserve_up,reserve_down,SDnoCD];
% penalty=[Deficit_response,Surplus_response];

% RealObjective = 0;
% for k=1:Horizon+Cepsillon*epsillon
%     RealObjective = DAUC_totalcost+sum(Upcost*RU+Downcost*RD);%RealObjective = RealObjective + Marginal_C*P(:,k)+(Upcost-Marginal_C)*RU(:,k)+(Marginal_C-Downcost)*RD(:,k)+Cnoload*state(:,k)+CSU*y(:,k)+CSD*z(:,k)+Cepsillon*epsillon(:,k)
% end
RealObjective=sum(Q*(P.*P)+Marginal_C*P)+sum(sum(fast_price.*Fast_respond,1))+sum(sum(shed_price.*deep_peak,1));%+Cepsillon*abs(Fast_respond)++350*ones(1,Nunits)*abs(Fast_respond)
Constraints_actual = [Minmaxpower_reserve,penalty,Actualgrid,Minmaxpower,Rampuplimit,Rampdownlimit];%
sol_actual=optimize(Constraints_actual,RealObjective,ops);%
if sol_actual.problem~=0
sol_actual.info
yalmiperror(sol_actual.problem)
end
Actualcost = value(RealObjective);
ActualP = value(P);
stac_sce_spec=stac_sce(cases,:);
Actual_Fast_respond = round(value(Fast_respond),4);
Actual_deep_peak=round(value(deep_peak),4);
Actual_energy_SD=value(energy_SD);
Actual_vSd=value(vSd);
Actual_vSc=value(vSc);
% fprintf('Actual cost is:%d \n the difference between actual and ideal is: %d \n',Actualcost,(Actualcost-idealcost));
Actual_sumload=sum(Actual_Load,1);
Actual_sumgene=sum(ActualP,1)+sum(Actual_Fast_respond,1)-sum(Actual_deep_peak,1);%
Actual_pchange=ActualP(:,2:end)-ActualP(:,1:end-1);
Actual_DACU=ActualP-DAUC_P;
Margin_actual=Q'.*ActualP+Marginal_C';
Totcost_actual=Q'.*(ActualP.*ActualP)+Marginal_C'.*ActualP;
% DFDF=sum(Totcost_actual(:,19))-sum(Totcost_IDEAL(:,19));
% Actual_sumfast=sum(Actual_Fast_respond,1);
% Actual_Flow_flowmargin=max(abs(Actual_Flow),[],2)-mpc.branch(:,6);
if all(sum(abs(ideal_sumload-Actual_sumgene))>=1e-2)
    warning('Real time schedule failed to satisfy demand')
end
Actual_segment_cost=[];
for t=1:Horizon
    Actual_segment_cost=[Actual_segment_cost,Q*(ActualP(:,t).*ActualP(:,t))+Marginal_C*ActualP(:,t)+sum(fast_price(:,t).*Actual_Fast_respond(:,t))+sum(shed_price(:,t).*Actual_deep_peak(:,t))];%+350*ones(1,Nunits)*abs(Actual_Fast_respond(:,t))+
end
% Actual_segment_cost1=[];
% for t=1:Horizon
%     Actual_segment_cost1=[Actual_segment_cost1,Q*(ActualP(:,t).*ActualP(:,t))+Marginal_C*ActualP(:,t)];%+350*ones(1,Nunits)*abs(Actual_Fast_respond(:,t))+
% end
% Actual_segment_cost2=[];
% for t=1:Horizon
%     Actual_segment_cost2=[Actual_segment_cost2,fast_price*Actual_Fast_respond(:,t)+shed_price*Actual_deep_peak(:,t)];%+350*ones(1,Nunits)*abs(Actual_Fast_respond(:,t))
% end
% pause
%% Store PCE & MPE respectively

PCE=[PCE;(Actual_segment_cost-ideal_segment_cost)./ideal_segment_cost];%./ideal_segment_cost
% PCE=[PCE;(Actual_segment_cost-ideal_segment_cost)];

MPE=[MPE;stac_sce(cases,:)-ones(1,Horizon)];%(Forecast_Load(:,:,cases)-Actual_Load)./Actual_Load
% MPE_ele=[MPE_ele;(Forecast_Load(:,:,cases)-Actual_Load)];%(Forecast_Load(:,:,cases)-Actual_Load)./Actual_Load

end
PCE = round(PCE,4);
[idx,idy]= find(PCE<0);
toc
% Afc1=Actual_fa(1:4:end,:)+Actual_pe(1:4:end,:);%BESS1
% Afc2=Actual_fa(2:4:end,:)+Actual_pe(2:4:end,:);%BESS2
% Afc3=Actual_fa(3:4:end,:)+Actual_pe(3:4:end,:);%BESS3
% Afc=Afc1+Afc2+Afc3;%%BESS_TOTAL

% %% Plot the cost function w.r.t forecasting errors
% path = 'F:\ETH learning material\Master Thesis\Code\Allplots\Allplots_new' ;   % mention your path 
% myfolder = '20200726' ;   % new folder name 
% folder = mkdir([path,filesep,myfolder]) ;
% path  = [path,filesep,myfolder] ;
B_aggregate = [];
PCE_aggregate = [];
% load('MPE.mat')
% load('PCE.mat')
% Horizon=24;

%%Plot and MPE RIZE
% MPE = squeeze(sum(reshape(MPE_ele',24,30,[]),2))';
for i= 1:Horizon
    fig=figure(i);
    [FEP,FEPC] = prepareCurveData(MPE(:,i),PCE(:,i));
    [x_FEP, sortOrder] = sort(FEP, 'ascend');
    y_FEPC=FEPC(sortOrder);
    [B,~,ib]=unique(x_FEP);%1 change to i
    indices2 = accumarray(ib, find(ib), [], @(rows){rows});
%     PCE_MEAN=zeros(size(indices2,1),1);
%     for k=1:size(indices2,1)
%         PCE_MEAN(k)=mean(y_FEPC(indices2{k}));%%1 change to i
%     end
    hold on
    options = fitoptions('Method','SmoothingSpline','SmoothingParam',0.99999);%,..0.001
%     options = fitoptions('poly6','Normalize','on');%,..0.001
% % %     fitVAR{i}=fit(B,round(PCE_MEAN,3),'smoothingspline',options);%    
    fitVAR{i}=fit(x_FEP,y_FEPC,'smoothingspline',options);%'poly6',options
    h=plot(fitVAR{i},x_FEP,y_FEPC);%     plot(B,PCE_MEAN,'-x');
   h(1).LineWidth = 3;
   h(1).Color='g';
    h(2).LineWidth = 1.5;
%     h(2).Color='blue';
%     scatter(MPE(:,i),PCE(:,i));
    hold off
%     title(['Smoothing spline for hour ' num2str(i)])
    legend('Original points','Fitting curve','Location',[0.62 0.8 0.1 0.1])       
    ax = gca;
    ax.XAxis.Exponent = 0;
    ax.YAxis.Exponent = 0;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
    axis([-0.1 0.1 -0.05 Inf])
    xlabel('FEP');
    dfy=ylabel('FEPC') ;
    set(dfy, 'position', get(dfy,'position')-[0.04,0,0]); 
%     B_aggregate(i,:)=B;
%     PCE_aggregate(i,:)=round(PCE_MEAN,3);
%     filename = sprintf('Loss%02d.jpeg',i);%%save and convert to jpeg
%       print( fig, '-djpeg', filename );
end


% for i=19:19
%     figure()
% %     scatter(MPE(:,i),Afc1(:,i),'b','o');
% %     hold on
% %     scatter(MPE(:,i),Afc2(:,i),'r','+');
% %     scatter(MPE(:,i),Afc3(:,i),'g','*');
% %     scatter(MPE(:,i),Afc(:,i),'k','d');
%     scatter(MPE(:,i),PCE_DAUC2(:,i),'r','d');
% %     hold off
% %     legend({'BESS(1)','BESS(2)','BESS(3)','BESS(tot)'},'Location',[0.26 0.8 0.1 0.1])
% %     title(['BESS dispatch for hour ' num2str(i)])
% %     dfy=ylabel('Power') ;
% end
%% Fit average of 24 hours into one loss function
[XOut,YOut] = prepareCurveData(B_aggregate,PCE_aggregate);
% fitVAR_syn=fit(XOut,YOut,'smoothingspline');
options = fitoptions('Method','SmoothingSpline','SmoothingParam',1e-9);%,... 0.9999999                 
fitVAR_syn=fit(XOut,YOut,'smoothingspline',options);%  
h=plot(fitVAR_syn,XOut,YOut);
%     plot(B,PCE_MEAN,'-x');
hold on
h(1).LineWidth = 3;
h(1).Color='g';
h(2).LineWidth = 1.5;
hold off
legend('Original points','Fitting curve','Location',[0.62 0.8 0.1 0.1])
title(['Smoothing spline for aggregated 24 hours'])
ax = gca;
ax.XAxis.Exponent = 0;
ax.YAxis.Exponent = 0;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
axis([-0.1 0.1 0 Inf])
dfx=xlabel('FEP');
dfy=ylabel('FEPC') ;
    dfy=ylabel('FEPC') ;
    set(dfy, 'position', get(dfy,'position')-[0.04,0,0]); 

%% Convert to .mat to .fig file % d=dir('*.fig'); % capture everything in the directory with FIG extension
% allNames={d.name}; % extract names of all FIG-files

% close all; % close any open figures
% for i=1:length(allNames)
%       open(allNames{i}); % open the FIG-file
%       base=strtok(allNames{i},'.'); % chop off the extension (".fig")
%       print('-djpeg',base); % export to JPEG as usual
%       close(gcf); % close it ("gcf" returns the handle to the current figure)
% end
%% Convert existing plots into Latex eps plot
figs = findobj(0, 'type', 'figure');
for k=1:length(figs)
    % print each figure in figs to a separate .eps file  
    print(figs(k), '-depsc', sprintf('Losspiece%d.eps', 25-k)) 
end
%% !!! Generate linear loss functions !!!
store_slope=zeros(Horizon,2);
for i= 1:Horizon
    fig=figure(i);
    [FEP,FEPC] = prepareCurveData(MPE(:,i),PCE(:,i));
    [x_FEP, sortOrder] = sort(FEP, 'ascend');
    y_FEPC=FEPC(sortOrder);
    FEP_neg=x_FEP(x_FEP<0);
    FEPC_neg=y_FEPC(x_FEP<0);
    FEP_pos=x_FEP(x_FEP>=0);
    FEPC_pos=y_FEPC(x_FEP>=0);
    fitVAR_neg{i}=fitlm(FEP_neg,FEPC_neg,'Intercept',false);
    fitVAR_pos{i}=fitlm(FEP_pos,FEPC_pos,'Intercept',false);
    store_slope(i,1)=fitVAR_neg{i}.Coefficients.Estimate;
    store_slope(i,2)=fitVAR_pos{i}.Coefficients.Estimate;
%     [B,~,ib]=unique(x_FEP);%1 change to i
%     indices2 = accumarray(ib, find(ib), [], @(rows){rows});
%     PCE_MEAN=zeros(size(indices2,1),1);
%     for k=1:size(indices2,1)
%         PCE_MEAN(k)=mean(y_FEPC(indices2{k}));%%1 change to i
%     end
    hold on
    plot(FEP_neg,store_slope(i,1)*FEP_neg);
    plot(FEP_pos,store_slope(i,2)*FEP_pos);
     scatter(MPE(:,i),PCE(:,i));
    hold off
%     options = fitoptions('Method','SmoothingSpline','SmoothingParam',0.999999);%,...
% %                     % 
% %                     %,...'Normalize','on');                     
% % %     fitVAR{i}=fit(B,round(PCE_MEAN,3),'smoothingspline',options);%    
%     fitVAR{i}=fit(x_FEP,y_FEPC,'smoothingspline',options);%,'SmoothingParam',0.07
%     h=plot(fitVAR{i},x_FEP,y_FEPC);%     plot(B,PCE_MEAN,'-x');
%    h(1).LineWidth = 3;
%    h(1).Color='g';
%     h(2).LineWidth = 1.5;
% %     h(2).Color='blue';
% %     scatter(MPE(:,i),PCE(:,i));
%     hold off
%     title(['Smoothing spline for hour ' num2str(i)])
%     legend('Original points','Fitting curve','Location',[0.62 0.8 0.1 0.1])       
    ax = gca;
    ax.XAxis.Exponent = 0;
    ax.YAxis.Exponent = 0;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
    axis([-0.1 0.1 -0.05 Inf])
    xlabel('FEP');
    dfy=ylabel('FEPC') ;
    set(dfy, 'position', get(dfy,'position')-[0.04,0,0]); 
%     filename = sprintf('Loss%02d.jpeg',i);%%save and convert to jpeg
%       print( fig, '-djpeg', filename );
end