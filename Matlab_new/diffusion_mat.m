function [df_mat] = diffusion_mat(data, epsilon, N)
%DIFFUSION_MAT 此处显示有关此函数的摘要
%   data:cell
df_mat = zeros(size(data,2));
for i =1:size(data,2)
    for j=i+1:size(data,2)
        df_mat(i ,j) = gaussian_kfunc(data{1,i},data{1,j},epsilon,N);
    end
end
df_mat = df_mat + df_mat' + eye(size(data,2));  % calculate the 'distance', same values result 1
end

function [data]=gaussian_kfunc(x1, x2, epsilon, N)
data=exp(-sum(abs(x1 -x2),'all')^2 /( 2 *epsilon *N^2));
end