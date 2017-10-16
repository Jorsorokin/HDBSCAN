function mr = mutual_reachability( X,coreDist )
% mr = mutual_reachability( X,coreDist )
%
% Finds the mutual reachability between points as:
%
%		max{ core(i),core(j),D(i,j) }
%
% Inputs:
%	X - n x m data matrix or n x n distance matrix
%	coreDist - 1 x n or n x 1 vector of core distances
%
% Outputs:
%	mr - n x n mutual reachability graph
%
% written by Jordan Sorokin, 10/7/2017

[n,m] = size( X );
if n ~= m
    mr = compute_pairwise_dist( X );
else
    mr = X;
end

% find where core(i) or core(j) > D(i,j)
maxCore = bsxfun( @max,coreDist,coreDist' );
idx = maxCore > mr; % n x n logical
mr(idx) = maxCore(idx); % ~idx locations left as D

end