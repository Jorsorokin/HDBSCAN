function [corePoints,coreLambda] = get_core_points( parents,bestClusters,lambdaMax )
    % [corePoints,coreLambda] = get_core_points( clusters,parents,bestClusters,lambdaMax ) 
    %
    % finds the core points for the final kept clusters by recursively 
    % searching for all subchildren of each kept cluster, and finding which
    % points persisted the longest (maximum lambda value)
    %
    % Inputs: 
    %   parents - 1 x K vector of parents of clusters 1:K
    %
    %   bestClusters - 1 x p <= K vector of kept clusters, as determined by
    %                  the function OFCH.m
    %
    %   lambdaMax - n x K matrix of lambda values for each point i and
    %               cluster j. LambdaMax is the lambda value at which point
    %               i "falls out of" cluster j
    %
    % Outputs:
    %   corePoints - 1 x p cell array of core points, with each cell representing
    %                one cluster ID from the list of best clusters
    %
    %   coreLambda - 1 x p vector of lambda max values associated with the
    %                p best clusters
    %
    % Written by Jordan Sorokin, 10/15/2017
    
    p = length( bestClusters );
    corePoints = cell( 1,p );
    coreLambda = zeros( 1,p );
    
    for k = 1:numel( bestClusters )
        thisClust = bestClusters(k);
        
        % find subnodes of bestCluster(k)
        children = get_all_subnodes( thisClust,parents );
        
        % get maximum lambda value of this cluster or its children
        maxLambda = max( lambdaMax(:,[thisClust,children]),[],2 ); % finds maximum lambda bewteen subnodes
        coreLambda(k) = max( maxLambda ); % finds the maximum lambda across all points and subnodes
        
        % find the core points of this cluster/subclusters
        corePoints{k} = find( maxLambda == coreLambda(k) );
    end
end
