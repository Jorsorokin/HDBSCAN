function score = GLOSH( clusters,parent,lastClust,lambdaNoise )
    % Global-Local Outlier Score from Hierarchy 
    %
    % score = GLOSH( delta,parent,lastClust,lambdaNoise )
    %
    % Computes the outlier score for each point 1:n given a 
    % cluster indicator vector "delta" and minimum edge weight 
    % at which each point i existed in the hierarchy
    %
    % Inputs
    %   clusters - 1 x K vector of cluster IDs
    %
    %   parent - 1 x K vector of parent clust for each cluster i
    %
    %   lastClust - 1 x n vector specifying the last cluster i that each
    %               point belonged to before becoming noise
    %
    %   lambdaNoise - 1 x n vector specifying the lambda value beyond which
    %                 point i becomes noise
    %
    % Output
    %   score - 1 x n vector of outlier scores, computed as:
    %   
    %               score(i) = 1 - refWeight / lastWeight(i)
    %
    %           where refWeight = min( lastWeight )
    % 
    % Written by Jordan Sorokin, 10/5/17
    
    score = zeros( size( lastClust ) );
    hasProcessed = false( 1,max( clusters ) );
    
    for clust = clusters
        
        % check if we've processed this cluster
        if hasProcessed(clust)
            continue
        end
        
        % perform depth-first search from this node to get children
        children = get_all_subnodes( clust,parent );

        % get epsilon when this cluster or any subclusters completely
        % disappear (when all X(i) == 0 for this branch)
        pts = ismember( lastClust,[clust,children] );
        refLambda = max( lambdaNoise(pts) );

        % compute the Global-Local Outlier Scores
        score(pts) = 1 - (lambdaNoise(pts) ./ refLambda); % [EQ 8]
        
        hasProcessed([clust,children]) = true;
    end
end