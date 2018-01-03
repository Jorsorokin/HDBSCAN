function [labels,P] = hdbscan_predict( X,model )
    % [labels,P] = hdbscan_predict( X,model )
    %
    % predicts cluster assignment for points in X based on 
    % a previous HDBSCAN clustering model
    %
    % Inputs:
    %	X - n x m data matrix, with n = points, m = dimensions. Note that 
    %		the # of columns of X must equal the # of columns of the original
    %		data used to create the clustering model
    %
    %	model - a structure created by a previous HDBSCAN run
    %
    % Outputs:
    %	labels - n x 1 vector of cluster assignment labels
    %
    %	P - n x 1 vector of probabilities, where P(i) is the probability
    %		of X(i) belonging to the cluster labels(i). In the case of outliers,
    %		labels(i) = 0, P(i) also equals 0.
    %
    % Written by Jordan Sorokin, 10/13/2017

	% check inputs
	[n,m] = size( X );
	[~,p] = size( model.corePoints );
	if m ~= p
		error( 'data in X must have same dimension as original data used for clustering!' );
    end
        
    % set variables 
    K = numel( model.clustID );
	avgDistances = zeros( n,1 );
	coreLabels = model.coreLabels;
    coreInd = model.coreInds;
    
	% compute the distances between X and the core points
	% Unless n is HUGE or we have an extremely large # of clusters,
	% this distance matrix shouldn't take up much memory
	D = compute_pairwise_dist( X,model.corePoints );

	% compute, for each cluster, the average distance between
	% point X(i) and dCore(j) for j in cluster k
	for k = 1:K
		avgDistances(:,k) = mean( D(:,coreLabels==k),2 );
	end

	% set labels for new points equal to the column that minimizes
	% the average distances to the core points
	[dists,labels] = min( avgDistances,[],2 );
    allLabels = unique( labels )';

	% calculate the probability that any point belongs to the cluster it 
	% was just assigned to by taking the ratio of the average of the
	% the core distances of cluster k with the average distance of point 
	% X(i) to the core points of cluster k
	P = zeros( n,1 );
	for k = allLabels
		ind = (labels == k);
        avgCoreDist = mean( model.dCore(coreInd(coreLabels == k)) );
		P(ind) = avgCoreDist ./ dists(ind);

		% set any label = 0 if the probabilities associated with points in this cluster
		% are < minimum of probabilities of original points belonging to this cluster
		minP = min( avgCoreDist ./ model.dCore(coreLabels == k) );
        outlier = P < minP;
		labels(outlier & ind) = 0;
    end
    
    labels = uint8( labels );
    P = single( P );
end
	

