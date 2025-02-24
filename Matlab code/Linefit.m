%% -------------------------Fitting and break points determination-----------------------------%
% rng(350);
% x=sort(-0.1+0.2*rand(20,1));
% y=2*abs(x).^2+exp(2*abs(x))+cos(0.1*2*pi*x)-2+0.01*rand(20,1);
% x=B;
% y=PCE_MEAN;
% fun=@(x)feval(fitVAR{i},x);
% value = integral(fun,min(fitVAR{i}.p.breaks),max(fitVAR{i}.p.breaks),'ArrayValued',true);
Horizon=24;
for i=19:19
tic
% coefficientNames = coeffvalues(fitVAR_syn);
x1=-0.1:1e-6:0.1;
% break_min=min(fitVAR{i}.p.breaks);
% break_max=max(fitVAR{i}.p.breaks);
% break_min=min(fitVAR_syn.p.breaks);
% break_max=max(fitVAR_syn.p.breaks);
% x1=break_min:1e-6:break_max;
% fit1=fit(x,y,'smoothingspline');
[d1,d2] = differentiate(fitVAR{i},x1);
% [d1,d2] = differentiate(fitVAR_syn,x1);
f_dotdot=abs(d2).^(2/5);
% f_max=max(abs(d2));
%% Calculate total integral within range
ix1 = find(x1 <= -0.1, 1, 'last');
ix2 = find(x1 >= 0.1, 1, 'first');
f_dotdotlim = f_dotdot(ix1:ix2);
x1lim = x1(ix1:ix2);
int_f_dotdot = trapz(x1lim, f_dotdotlim);
%% Quantify the fitting errors
Linear_error=[];
for nots=1:100
Linear_error(nots)=int_f_dotdot^(5/2)/(nots^2*sqrt(120));%%f_max*(0.2)^(5/2)/(nots^2*sqrt(120));approximate for 100 node cases and find the most small N requirement
end
N_knots=find(Linear_error<=1e-3,1,'first');
% figure()
% ada=plot(1:100,Linear_error);
% set(ada,'LineWidth',2)
% legend('Upper bound of approximation error','Location','northeast','Orientation','horizontal')
% xlabel('Number of breakpoints');
% ylabel('Error') ;
% title('Approximation error of linear interpolant')
%% Numerate all discrete point integral
nonminaotor=[];
for x_i=-0.1+1e-7:1e-6:0.1
    ixa = find(x1 <= -0.1, 1, 'last');
    ixx = find(x1 >= x_i, 1, 'first');
    f_dotdotund = f_dotdot(ixa:ixx);
    xiilim = x1(ixa:ixx);
    nonminaotor=[nonminaotor;trapz(xiilim, f_dotdotund)];
end
F_x=nonminaotor./int_f_dotdot;
index_result={};

%% Find the index which satisfy condition 17 & 18
for k=1:N_knots
    index_result{k}=find(abs(F_x-k/N_knots)<=1e-4)';
end
% Take only one element within the cells
final_ind= cellfun(@(v)v(1),index_result);
x11=-0.1+1e-7:1e-6:0.1;% should be identical to x_i

x_cordin_raw=x11(final_ind)';
y_cordin_raw=feval(fitVAR{i},x_cordin_raw);
[x_cordin_1, unique_index] = unique(x_cordin_raw, 'stable' );
y_cordin_1=y_cordin_raw(unique_index);
x_cordin_neg=x_cordin_1(x_cordin_1<0);
x_cordin_pos=x_cordin_1(x_cordin_1>=0);
y_cordin_neg=y_cordin_1(x_cordin_1<0);
y_cordin_pos=y_cordin_1(x_cordin_1>=0);
wrong_neg=find(diff(y_cordin_neg)>0)+1;
wrong_pos=find(diff(y_cordin_pos)<0)+1;
x_cordin_neg(wrong_neg)=[];
y_cordin_neg(wrong_neg)=[];
x_cordin_pos(wrong_pos)=[];
y_cordin_pos(wrong_pos)=[];
x_cordin{i}=[x_cordin_neg;x_cordin_pos];
y_cordin_zero=[y_cordin_neg;y_cordin_pos];
y_cordin_zero(y_cordin_zero<0)=0;
y_cordin{i}=y_cordin_zero;

% x_cordin_syn_raw=x11(final_ind)';
% y_cordin_syn_raw=feval(fitVAR_syn,x_cordin_syn_raw);
% [x_cordin_syn_1, unique_index] = unique(x_cordin_syn_raw, 'stable' );
% y_cordin_syn_1=y_cordin_syn_raw(unique_index);
% x_cordin_syn_neg=x_cordin_syn_1(x_cordin_syn_1<0);
% x_cordin_syn_pos=x_cordin_syn_1(x_cordin_syn_1>=0);
% y_cordin_syn_neg=y_cordin_syn_1(x_cordin_syn_1<0);
% y_cordin_syn_pos=y_cordin_syn_1(x_cordin_syn_1>=0);
% wrong_neg=find(diff(y_cordin_syn_neg)>0)+1;
% wrong_pos=find(diff(y_cordin_syn_pos)<0)+1;
% x_cordin_syn_neg(wrong_neg)=[];
% y_cordin_syn_neg(wrong_neg)=[];
% x_cordin_syn_pos(wrong_pos)=[];
% y_cordin_syn_pos(wrong_pos)=[];
% x_cordin_syn=[x_cordin_syn_neg;x_cordin_syn_pos];
% y_cordin_syn=[y_cordin_syn_neg;y_cordin_syn_pos];
toc
end
% x_cordin_syn=x_cordin{i};
% y_cordin_syn=y_cordin{i};
% fitVAR_syn=fitVAR{i};
%% check the size of cells inside the cell array
cellsz = cellfun(@size,x_cordin,'uni',false);
%% Plot results 
for i=1:Horizon
    figure(i)
    scatter(x_cordin{i},y_cordin{i},60,'filled','d','k');
end
%%
subplot(2,1,1)
F_plot=plot(-0.1+1e-7:1e-6:0.1,F_x);
set(F_plot,'LineWidth',2)
xlim([-0.1 0.1])
hY=ylabel('F(\epsilon)','fontsize',12) ;
get(hY,'interpreter')
% xlabel('MPE')
hold on
for k=1:N_knots
dash_plot1=plot([-1 x_cordin_syn(k)], [k/N_knots k/N_knots],'--','Color',[0.9290 0.6940 0.1250]);
dash_plot2=plot([x_cordin_syn(k) x_cordin_syn(k)], [0 k/N_knots],'--','Color','g');
set(dash_plot1,'LineWidth',1.2)
set(dash_plot2,'LineWidth',1.2)
end
hold off

subplot(2,1,2)
% % plot(x1,feval(fitVAR{i},x1)) % cfit plot method

% scatter(x,y,'r')
% plot(fitVAR{i});
h(1)=plot(fitVAR_syn);
set(h(1),'LineWidth',1.2)
hold on
% scatter(x_cordin{i},y_cordin{i},60,'filled','d','k')
h(2)=scatter(x_cordin_syn,y_cordin_syn,50,'filled','d','k');
for k=1:N_knots
dash_plot3=plot([x_cordin_syn(k) x_cordin_syn(k)], [0 y_cordin_syn(k)],'--','Color','g');
set(dash_plot3,'LineWidth',1.2)
end
legend(h([1 2]),'Smoothing spline','Breakpoints','Location','northeast')
hold off
% title('Linear interpolation with determined breakpoints') 
% ax = gca;
% ax.XAxisLocation = 'origin';
% ax.YAxisLocation = 'origin';
axis([-0.1 0.1 -0.005 0.65])
hXL=xlabel('\epsilon','fontsize',12);
hYL=ylabel('{s(\epsilon)}','fontsize',12) ;
get(hXL,'interpreter')
get(hYL,'interpreter')

figs = findobj(0, 'type', 'figure');
for k=1:length(figs)
    % print each figure in figs to a separate .eps file  
    print(figs(k), '-depsc', sprintf('Partition.eps')) 
end
% subplot(3,1,2)

% plot(x1,d1,'m') % double plot method
% grid on
% legend('1st derivative')
% title('First derivative of Spline interpolation') 
% xlabel('MPE');
% subplot(3,1,3)
figure()
w1w=plot(x1,f_dotdot,'c','Color',[0.8500 0.3250 0.0980]); % double plot method
set(w1w,'LineWidth',1.2)
grid on
% legend('2nd derivative of approximation function to the power of 0.4')
% title('Local knot density') 
% str = '$$ |f''(x)|^\frac{2}{5} $$';
% kkkk=text(1.1,0.5,str,'Interpreter','latex');
% legend('Derivative of approximation function')
xlabel('x');
% ylabel('f''(x)');
% end
for k=1:Horizon
y_cordin{k}=round(y_cordin{k},2);
end
%% Convert to Matrix with 0 appended (for Python training)
x_cordin_syn=x_cordin_syn';
y_cordin_syn=round(y_cordin_syn,2);
y_cordin = cellfun(@transpose,y_cordin,'UniformOutput',false);
cellsx = cellfun(@size,x_cordin,'uni',false);
cellsy = cellfun(@size,y_cordin,'uni',false);
ML_x = max(cellfun(@numel, x_cordin'));
X_out = cellfun(@(x) [x zeros(1, ML_x - numel(x))], x_cordin', 'un', 0);
x_cor=cell2mat(X_out);
ML_y = max(cellfun(@numel, y_cordin'));
Y_out = cellfun(@(x) [x zeros(1, ML_y - numel(x))], y_cordin', 'un', 0);
y_cor=cell2mat(Y_out);
x_cor=round(x_cor,3);
y_cor=round(y_cor,3);



% for i=1:Horizon
a=[round(x_cordin{1},3),round(y_cordin{1},3)];
b=[round(x_cordin{2},3),round(y_cordin{2},3)];
c=[round(x_cordin{3},3),round(y_cordin{3},3)];
d=[round(x_cordin{4},3),round(y_cordin{4},3)];
e=[round(x_cordin{5},3),round(y_cordin{5},3)];
f=[round(x_cordin{6},3),round(y_cordin{6},3)];
g=[round(x_cordin{7},3),round(y_cordin{7},3)];
h=[round(x_cordin{8},3),round(y_cordin{8},3)];
i=[round(x_cordin{9},3),round(y_cordin{9},3)];
j=[round(x_cordin{10},3),round(y_cordin{10},3)];
k=[round(x_cordin{11},3),round(y_cordin{11},3)];
l=[round(x_cordin{12},3),round(y_cordin{12},3)];
m=[round(x_cordin{13},3),round(y_cordin{13},3)];
n=[round(x_cordin{14},3),round(y_cordin{14},3)];
o=[round(x_cordin{15},3),round(y_cordin{15},3)];
p=[round(x_cordin{16},3),round(y_cordin{16},3)];
q=[round(x_cordin{17},3),round(y_cordin{17},3)];
r=[round(x_cordin{18},3),round(y_cordin{18},3)];
s=[round(x_cordin{19},3),round(y_cordin{19},3)];
t=[round(x_cordin{20},3),round(y_cordin{20},3)];
u=[round(x_cordin{21},3),round(y_cordin{21},3)];
v=[round(x_cordin{22},3),round(y_cordin{22},3)];
w=[round(x_cordin{23},3),round(y_cordin{23},3)];
x=[round(x_cordin{24},3),round(y_cordin{24},3)];

% end
break_point24=break_point';

%% Plot diagram into eps
figs = findobj(0, 'type', 'figure');
for k=1:length(figs)
    % print each figure in figs to a separate .eps file  
    print(figs(k), '-depsc', sprintf('Losscompare%d.eps', 25-k)) 
end
%% Put loss into table form and plot it
load('Horizon.mat')
breakpoint = readtable('E:/ETH learning material/Master Thesis/Code/Newplots/b0.csv');
breakpoint_sn = readtable('E:/ETH learning material/Master Thesis//Code/Newplots/b1.csv');
losssum=readmatrix('E:/ETH learning material/Master Thesis/Code/Newplots/losssum.csv');
error_dist=readmatrix('E:/ETH learning material/Master Thesis/Code/Newplots/error_dist.csv');
losssum_mse=readmatrix('E:/ETH learning material/Master Thesis/Code/Newplots/losssum_mse.csv');
error_mse=readmatrix('E:/ETH learning material/Master Thesis/Code/Newplots/error_dist_mse.csv');
error_hourly=[];
error_hourly_mse=[];
for i= 1:size(losssum,2)
   error_hourly=[error_hourly,error_dist(i:24:end)]; 
end
for i= 1:size(losssum_mse,2)
   error_hourly_mse=[error_hourly_mse,error_mse(i:24:end)]; 
end
% knot_chose=rmmissing(breakpoint{:,1:2});
x_p=0:0.001:0.1;
x_n=-0.1:0.001:0;
x_tot=-0.1:0.0001:0.1;


for i= 1:Horizon
    figure(i)
    y_tot=[];
    y_tot_sn=[];
    for j=1:length(x_tot)
        y_tot=[y_tot,piece(rmmissing(breakpoint{:,(2*i-1):2*i}),x_tot(j))];
    end
    for j=1:length(x_tot)
        y_tot_sn=[y_tot_sn,piece(rmmissing(breakpoint_sn{:,1:2}),x_tot(j))];
    end
    hold on
    lin1=plot(x_tot,y_tot,'k');
    scatter(error_hourly(:,i),losssum(:,i),20,'r','+');
    scatter(error_hourly_mse(:,i),losssum_mse(:,i),10,'b','o');
%     plot(x_tot,y_tot_sn,'--');
%     lin1=plot(x_p,pos_coe(i)*x_p,'--g');
%     lin2=plot(x_n,neg_coe(i)*x_n,'--g');
%     scatter(MPE(:,i),PCE(:,i));
%     set(get(get(lin2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(lin1,'LineWidth',2);
%     set(lin2,'LineWidth',1.2);
     legend({'Loss function','hourly','mse'},'Location',[0.26 0.78 0.1 0.1]);%,'Linear approximation','Original points',[0.26 0.8 0.1 0.1]
    hold off
    title(['Hour ' num2str(i)])
    ax = gca;
    ax.XAxis.Exponent = 0;
    ax.YAxis.Exponent = 0;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
%     axis([-Inf Inf 0 1])
%     xlabel('FEP');
%     dfy=ylabel('FEPC') ;
%     set(dfy, 'position', get(dfy,'position')-[0.03,0,0]); 
end


%% Plot three loss function
x_p=0:0.001:0.1;
x_n=-0.1:0.001:0;
x_tot=-0.1:0.0001:0.1;
load('Horizon.mat')
breakpoint = load('E:/ETH learning material/Master Thesis/Code/Allplots/Allplots_new/20200906/breakpoint_plots.mat').breakpoint;
breakpoint_sn = readtable('E:/ETH learning material/Master Thesis//Code/Newplots/b1.csv');
store_slope = load('E:/ETH learning material/Master Thesis/Code/Allplots/Allplots_new/20200906/store_slope.mat').store_slope;
PCE = load('E:/ETH learning material/Master Thesis/Code/Allplots/Allplots_new/20200906/PCE.mat').PCE;
MPE = load('E:/ETH learning material/Master Thesis/Code/Allplots/Allplots_new/20200906/MPE.mat').MPE;
x_break=breakpoint(:,1:2:end);
y_break=breakpoint(:,2:2:end);
for i= 1:Horizon
    figure(i)
    y_tot=[];
    y_tot_sn=[];
    for j=1:length(x_tot)
        y_tot=[y_tot,piece(rmmissing(breakpoint{:,(2*i-1):2*i}),x_tot(j))];
    end
    for j=1:length(x_tot)
        y_tot_sn=[y_tot_sn,piece(rmmissing(breakpoint_sn{:,1:2}),x_tot(j))];
    end
    hold on
    scatter(MPE(:,i),PCE(:,i),15,'g','.');
    h1=plot(x_tot,y_tot,'k');
    h2=plot(x_tot,y_tot_sn,'r--');
    lin1=plot(x_p,store_slope(i,2)*x_p,'--b');
    lin2=plot(x_n,store_slope(i,1)*x_n,'--b');
    set(get(get(lin2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(h1,'LineWidth',1.8);
    set(h2,'LineWidth',1.8);
    set(lin1,'LineWidth',1.2);
    set(lin2,'LineWidth',1.2);   
    legend({'Loss data','Hourly','Daily','Linear'},'Location',[0.7 0.78 0.1 0.1]);%[0.26 0.78 0.1 0.1]
    hold off
%     title(['Three loss functions in hour ' num2str(i)])
    ax = gca;
    ax.XAxis.Exponent = 0;
    ax.YAxis.Exponent = 0;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
    axis([-0.1 0.1 0 Inf])
    xlabel('FEP');
    dfy=ylabel('FEPC') ;
%     set(dfy, 'position', get(dfy,'position')-[0.03,0,0]); 
end
%% Plot the table results
mlr_result = readtable('E:/ETH learning material/Master Thesis/Code/Allplots/Allplots_new/20200906/mlr_results.xlsx');
ann_result = readtable('E:/ETH learning material/Master Thesis/Code/Allplots/Allplots_new/20200906/ann_results.xlsx');

hold on
FEPC_mse=plot(1:24,ann_result{1,2:end},'-+','Color',[0.4660 0.6740 0.1880],'LineWidth',2);
FEPC_linear=plot(1:24,ann_result{2,2:end},'-d','Color',[0.8500 0.3250 0.0980],'LineWidth',2);
FEPC_daily=plot(1:24,ann_result{3,2:end},'-s','Color',[0.9290 0.6940 0.1250],'LineWidth',2);
FEPC_hourly=plot(1:24,ann_result{4,2:end},'-o','Color',[0 0.4470 0.7410],'LineWidth',2);
hold off
legend({'MSE','Linear','Daily','Hourly'},'Location',[0.8 0.8 0.1 0.1]);
grid on
axis([0.5 24 -Inf Inf])
xlabel('Hour');
ylabel('MFEPC');


hold on
mape_mse=plot(1:24,mlr_result{5,2:end},'-+','Color',[0.4660 0.6740 0.1880],'LineWidth',2);
mape_linear=plot(1:24,mlr_result{6,2:end},'-d','Color',[0.8500 0.3250 0.0980],'LineWidth',2);
mape_daily=plot(1:24,mlr_result{7,2:end},'-s','Color',[0.9290 0.6940 0.1250],'LineWidth',2);
mape_hourly=plot(1:24,mlr_result{8,2:end},'-o','Color',[0 0.4470 0.7410],'LineWidth',2);
hold off
legend({'MSE','Linear','Daily','Hourly'},'Location',[0.8 0.8 0.1 0.1]);
grid on
axis([0.5 24 -Inf Inf])
xlabel('Hour');
ylabel('MAPE');

hold on
OFP_mse=plot(1:24,ann_result{9,2:end},'-+','Color',[0.4660 0.6740 0.1880],'LineWidth',2);
OFP_linear=plot(1:24,ann_result{10,2:end},'-d','Color',[0.8500 0.3250 0.0980],'LineWidth',2);
OFP_daily=plot(1:24,ann_result{11,2:end},'-s','Color',[0.9290 0.6940 0.1250],'LineWidth',2);
OFP_hourly=plot(1:24,ann_result{12,2:end},'-o','Color',[0 0.4470 0.7410],'LineWidth',2);
dash_plot=plot([0 24], [50 50],'--','Color','k','LineWidth',2);
hold off
legend({'MSE','Linear','Daily','Hourly'},'Location',[0.8 0.8 0.1 0.1]);%[0.7 0.8 0.1 0.1]
grid on
axis([0.5 24 20 80])
xlabel('Hour');
ylabel('OFP');

hold on
UFP_mse=plot(1:24,ann_result{13,2:end},'-+','Color',[0.4660 0.6740 0.1880],'LineWidth',2);
UFP_linear=plot(1:24,ann_result{14,2:end},'-d','Color',[0.8500 0.3250 0.0980],'LineWidth',2);
UFP_daily=plot(1:24,ann_result{15,2:end},'-s','Color',[0.9290 0.6940 0.1250],'LineWidth',2);
UFP_hourly=plot(1:24,ann_result{16,2:end},'-o','Color',[0 0.4470 0.7410],'LineWidth',2);
dash_plot=plot([0 24], [50 50],'--','Color','k','LineWidth',2);
hold off
legend({'MSE','Linear','Daily','Hourly'},'Location',[0.81 0.82 0.1 0.1]);
grid on
axis([0.5 24 20 80])
xlabel('Hour');
ylabel('UFP');

%% Plot all figures into eps format for Overleaf
% figs = findobj(0, 'type', 'figure');
% for k=1:length(figs)
%     % print each figure in figs to a separate .eps file  
%     print(figs(k), '-depsc', sprintf('Losscomparet%d.eps',25-k)) 
% end

figs = findobj(0, 'type', 'figure');
for k=1:length(figs)
    % print each figure in figs to a separate .eps file  
    print(figs(k), '-depsc', sprintf('MFEPC.eps')) 
end

figs = findobj(0, 'type', 'figure');
for k=1:length(figs)
    % print each figure in figs to a separate .eps file  
    print(figs(k), '-depsc', sprintf('MAPE.eps')) 
end

figs = findobj(0, 'type', 'figure');
for k=1:length(figs)
    % print each figure in figs to a separate .eps file  
    print(figs(k), '-depsc', sprintf('OFP.eps')) 
end

figs = findobj(0, 'type', 'figure');
for k=1:length(figs)
    % print each figure in figs to a separate .eps file  
    print(figs(k), '-depsc', sprintf('UFP.eps')) 
end
%% Density plot
x = readmatrix('F:/ETH learning material/Master Thesis/hist_syn_nn.csv');
figure();
h = histogram(x,'Normalization','probability');
h.NumBins = 30;
% hold on;
line([0, 0], [0,0.16], 'LineWidth', 2, 'Color', 'r');
% h.BinWidth = 0.005;
xlim([-1, 1]);
% ylim([0, 0.14]);
ylabel('Frequency');
xlabel('FEP');



figs = findobj(0, 'type', 'figure');
for k=1:length(figs)
    % print each figure in figs to a separate .eps file  
    print(figs(k), '-depsc', sprintf('Combine%d.eps',25-k)) 
end