function [labels,P] = get_cluster_probability( clusters,lambdaMax,coreLambda )
    % [labels,P] = get_cluster_probability( clusters,lambdaMax,coreLambda )
    %
    % computes the probability of point i belonging to cluster j, if 
    % point i is associated with cluster j (if lambdaMax(i,j) > 0)
    %
    % Inputs:
    %   clusters - 1 x K vector of cluster IDs
    %
    %   lambdaMax - n x P >= K matrix of lambda values of point i for
    %               cluster j.  
    %
    %   coreLambda - 1 x K vector of maximum lambda associated with cluster
    %                i or any of its subclusters
    %
    % Outputs:
    %   labels - n x 1 vector of cluster assignments
    %
    %   P - n x 1 vector of assignment probabilities
    %
    % Written by Jordan Sorokin, 10/15/2017

    n = size( lambdaMax,1 );
    P = zeros( n,1 );
    labels = zeros( n,1,'uint8' );
    
    % loop over clusters
    for k = 1:numel( clusters )
        pts = lambdaMax(:,clusters(k)) > 0;
        labels(pts) = k;
        P(pts) = lambdaMax(pts,clusters(k)) / coreLambda(k);
    end
end
