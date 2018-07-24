function [start,stop] = find_blockIndex_range( n,m,maxSize )
% [start,stop] = find_indexing_range( n,m,(maxSize) )
%
% find the indexing ranges to perform block looping over points "1:m" for 
% nearest-neighbor, etc. This greatly improves performance (not looping over 
% all points) while avoiding massive (> 1 GB) matrices. Default "maxSize",
% which refers to the maximum matrix size allowed, is 100,000,000 entries.
%
% "n" refers to number of elements in first vector, "m" refers to number of
% elements in second. This is more flexible than simply providing one
% number, as neighborhood graphs etc. may not necessarily be square.

if nargin < 3
    maxSize = 1e8; % 1 GB
end

maxPts = ceil( maxSize / n );
remainder = mod( m,maxPts );
start = 1:maxPts:m;
stop = maxPts:maxPts:m-remainder;
if remainder > 0
    stop = [stop,m];
end

end