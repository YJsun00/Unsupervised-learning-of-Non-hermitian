function [p_list] = construct_p_list(GAMMA, t1_list, T2, N)
%CONSTRUCT_P_LIST 此处显示有关此函数的摘要
%   此处显示详细说明
p_list = cell(1, length(t1_list));
index = 1;
count = 0;
for t1_init = t1_list
    [initial_P,count] = construct_p(GAMMA, t1_init, T2, N,count);
    modified_P = modify(initial_P);
    p_list{index}=modified_P;
    index = index+1;
end
end

function [p, count] = construct_p(gamma, t1, t2, N, count)
% Hamiltonian
U = [0 t1+gamma; t1-gamma 0];
T = [0 0; t2 0];
init = U;
for cell=1:N-1
    init = blkdiag(init, U);
end
for row=1:N-1
    init(2*row+1:2*row+2, 2*row-1:2*row) = T';
    init(2*row-1:2*row, 2*row+1:2*row+2) = T;
end

% Sort
[right_v, w1] = eig(init);  % w: eigenvalue; v: eigenstate. Complex Hermitian (conjugate symmetric) or a real symmetric matrix.
%[left_v, w2] = eig(init');
%w2 = conj(w2);
%w2 = diag(w2);
%[w2,inv_index] = sort(w2);
%left_v = left_v(:, inv_index);
w1 = diag(w1);
left_v=inv(right_v)';
%[w1, inv_index] = sort(w1);
%right_v = right_v(:, inv_index);

% construct projection matrix
index1 = find(real(w1)<0);
w1 = w1(index1);
right_v = right_v(:, index1);
left_v = left_v(:, index1);
p = right_v*transpose(conj(left_v));

end


function [modified_p]=modify(initial_P)
% modified_p = zeros(1, size(initial_P, 1)/2);
index = 1;
for row=1:2:size(initial_P, 1)
    modified_p(index)=initial_P(row, row+1);
    index = index+1;
end
end

