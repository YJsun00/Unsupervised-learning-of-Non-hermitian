clc
clear
format long;
% parameter
GAMMA = 0.4;
T2 = 0.6;
N = 40;
epsilon = 0.001;

t1_list = 0:1/pi^5:1.6306;
P_list = construct_p_list(GAMMA, t1_list, T2, N);
df_mat = diffusion_mat(P_list, epsilon, N);  % first time diffusion
figure(1);
imagesc(df_mat)
% h = heatmap(df_mat,'CellLabelColor','none');
% h.GridVisible = 'off';
% h.Colormap=summer;
df_prob = df_mat ./ sum(df_mat, 2);
[evs, evals] = eig(df_prob);
evals = diag(evals);
[evals,index] = sort(real(evals),'descend');
evs = evs(:, index);
figure(2); % First 20 eigen value
scatter(1:20, evals(1:20));
figure(3);  %eigenvector
subplot(2,2,1); 
scatter(t1_list, evs(:,1));
subplot(2,2,2);
scatter(t1_list, evs(:,2));
subplot(2,2,3);
scatter(t1_list, evs(:,3));
subplot(2,2,4);
scatter(t1_list, evs(:,4));

