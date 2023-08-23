%% Parameter setting
Nunits = size(mpc.gen,1);
Horizon = 24;
%Initial power output
Pini=[200;200;500;30;20;20];%22;132;66
% Power limit
Pmax=[300;700;600;300;200;200];
Pmin=Pmax*0.10;%[0;0;0;0;0;0];
% Ramping limit
Ramup=[90;130;90;120;120;120]';%Pmax'*0.3;%
Ramdown=[90;130;90;120;120;120]';%[90;144;160;160;160;160]';
% Marginal production cost
Q = [0.8,0.4,0.134,3.9,2.75,2.65]*0.05;%*0.01*0.05*1e4;
Marginal_C = [2,1.75,1,3.25,3,2.5]*0.1;%*1e4;
% Marginal_C = [30,15,10,45,40,35];
%%Conventional units incidence matrix: Location of Conventional units in electricity network
Unit2Node = zeros(size(mpc.gen,1),size(mpc.bus,1));
for i=1:size(mpc.gen,1)
Unit2Node(i,mpc.gen(i,1))=1;
end
%%Network lines incidence matrix: Start and End Node of each transmission line
branch_dir=mpc.branch(:,1:2);
Line2Node=[];
for k=1:size(mpc.branch,1)
    Line2 = zeros(size(mpc.bus,1),1)';
    Line2(branch_dir(k,1))=1;
    Line2(branch_dir(k,2))=-1;
    Line2Node = [Line2Node;Line2];%Line x Nodes
end
%%Storage to node
% syms er
% figure()
% y1=Q(1)*(er.*er)+Marginal_C(1)*er;
% y2=Q(2)*(er.*er)+Marginal_C(2)*er;
% y3=Q(3)*(er.*er)+Marginal_C(3)*er;
% y4=Q(4)*(er.*er)+Marginal_C(4)*er;
% y5=Q(5)*(er.*er)+Marginal_C(5)*er;
% y6=Q(6)*(er.*er)+Marginal_C(6)*er;
% fplot(y1,[0,Pmax(1)])
% hold on
% fplot(y2,[0,Pmax(2)])
% fplot(y3,[0,Pmax(3)])
% fplot(y4,[0,Pmax(4)])
% fplot(y5,[0,Pmax(5)])
% fplot(y6,[0,Pmax(6)])
% hold off
% legend('y1','y2','y3','y4','y5','y6')
% 
% Cost1=Q(1)*er+Marginal_C(1);
% Cost2=Q(2)*er+Marginal_C(2);
% Cost3=Q(3)*er+Marginal_C(3);
% Cost4=Q(4)*er+Marginal_C(4);
% Cost5=Q(5)*er+Marginal_C(5);
% Cost6=Q(6)*er+Marginal_C(6);
% fplot(Cost1,[0,Pmax(1)])
% hold on
% fplot(Cost2,[0,Pmax(2)])
% fplot(Cost3,[0,Pmax(3)])
% fplot(Cost4,[0,Pmax(4)])
% fplot(Cost5,[0,Pmax(5)])
% fplot(Cost6,[0,Pmax(6)])
% hold off
% legend('Cost1','Cost2','Cost3','Cost4','Cost5','Cost6')
 




