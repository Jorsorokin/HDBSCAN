classdef HDBSCAN < handle
    % clusterer = HDBSCAN( X )
    %
    % Calling HDBSCAN creates an instance of the HDBSCAN cluster object.
    %
    % HDBSCAN stands for: Hierarchical Density-Based Spatial Clustering, 
    % with Application with Noise. It is extensively described in:
    %   Campello et al. 2013 and Campello et al. 2015
    %
    % The HDBSCAN cluster object contains methods for training a hierarchical
    % clustering model based on the input data matrix X. Clustering is
    % performed by iteratively removing links between a
    % graph-representation of the original data (based on pairwise
    % distances), and searching for the resulting connected
    % components at each iteration. 
    %
    % This differs from other hierarchical clustering methods,
    % such as single linkage clustering, as clusters with less
    % than a minimum # of points are deemed noise. Additionally, following
    % the creation of the cluster hierarchy, an optimal, flat clustering is
    % performed based on the stability of each cluster. This gives a final 
    % clustering that can be performed at varying heights for each branch
    % of the cluster tree, which differs than DBSCAN which produces only a
    % single, horizontal cut through the tree.
    %
    % After training a model, one can also predict cluster membership for new
    % points not originally used in the model creation. Note that this gives an 
    % "approximate membership", as new points may have changed the model
    % hierarchy if used in the training procedure.
    %
    % Properties:
    % ----------
    %   data            -   the raw data used for model creation
    %   
    %   minpts          -   the nearest 'minpts' neighbor used for core distance
    %                       calculation for each point in X. Default = 5
    %   
    %   minclustsize    -   the minimum # of points necessary for a cluster
    %                       to be deemed valid. Default = 5
    %
    %   outlierThresh   -   a cutoff value between [0,1], where any X(i) with an outlier score 
    %                       (see below) greather than 'outlierThresh' is assigned
    %                       as an outlier (ID = 0). Default = 0.9
    %
    %   kdtree          -   a KD tree based on the data in X if the
    %                       dimensionality of X is <= 10
    %
    %   model           -   a trained hierarchical model. For details, see
    %                       'hdbscan_fit.m'.
    %
    %   labels          -   vector of cluster membership for each point i
    %
    %   bestClusters    -   the optimal clusters discovered from the clusterTree
    %
    %   corePoints      -   the most "representative" points of the final
    %                       optimal clusters. These are the densest points
    %                       of any of the clusters, and can be used for
    %                       predicting new data cluster membership
    %
    %   coreLambda      -   the lambda value associated with the core
    %                       points for each best cluster
    %
    %   score           -   the outlier score for each point i
    %
    %   dCore           -   the core distances of the points in X, given
    %                       the specified 'minpts'
    %
    %   P               -   probability of point i belonging to the cluster
    %                       indicated by labels(i)
    %
    % Methods:
    % -------
    %   fit             -   fits a hierarchical model to the data in X
    %
    %   predict         -   predicts new data based on the trained model
    %
    %   get_best_clusters - finds the optimal flat clustering from the
    %                       full cluster_tree hierarchy
    %
    %   get_membership  -   assigns a label and probability to each point in X 
    %                       based on the optimal flat clustering.
    %
    %   plot_tree       -   plots the cluster hierarchy, and indicates
    %                       which clusters were kept in the final clustering
    %
    %       * see individual methods for more details on inputs/outputs
    %
    % Written by Jordan Sorokin
    % 10/15/2017
    
    properties
        nPoints
        nDims
        model
        kdtree
        data
        minpts = 5;
        minclustsize = 5;
        outlierThresh = 0.9;
        bestClusters
        corePoints
        coreLambda
        labels
        score
        P
    end
    
    methods
        
        function self = HDBSCAN( X )
            % creates an instance of the HDBSCAN object
            
            self.data = X;
            self.nPoints = size( X,1 );
            self.nDims = size( X,2 );
        end
            
        
        function fit( self,varargin )
            % fit( self,(dEps,verbose) )
            %
            % fits a full hierarchical cluster model to the data stored in 
            % self.data. Uses "self.minpts" and "self.minclustsize" for
            % training the model. 
            %
            % Inputs:
            %   self - an instance of the HDSBCAN object
            %
            %   dEps - a scalar that specifies the number of iterations to
            %          do, as:
            %           
            %               nIters = iterations(1:dEps:end)
            %
            %           Larger values of 'dEps' results in faster model
            %           training, at the risk of more approximate cluster
            %           identification (default = 1)
            %   
            %   verbose - logical. prints clustering information if true
            %
            % Outputs:
            %   self.model
            
            % check inputs
            if nargin > 1 && ~isempty( varargin{1} )
                dEps = round( varargin{1} ) * sign( varargin{1} ); % ensures it's a positive integer
            else
                dEps = 1;
            end
            
            if nargin > 2 && ~isempty( varargin{2} )
                verbose = varargin{2};
            else
                verbose = true;
            end
            
            % remove previous cluster-based post processing
            self.bestClusters = [];
            self.corePoints = [];
            self.coreLambda = [];
            self.P = [];
            self.score = [];
            self.labels = [];
            
            % report cluster params if verbose = true
            if verbose
                fprintf( 'Training cluster hierarchy...\n' );
                fprintf( '\tData matrix size:\n' );
                fprintf( '\t\t%i points x %i dimensions\n\n',self.nPoints,self.nDims );
                fprintf( '\tMin # neighbors: %i\n',self.minpts );
                fprintf( '\tMin cluster size: %i\n',self.minclustsize );
                fprintf( '\tSkipping every %i iteration\n\n',dEps-1 );
                start = clock;
            end
                
            % fit the hierarchical cluster tree
            self.model = hdbscan_fit( self.data,...
                                'minpts',self.minpts,...
                                'minclustsize',self.minclustsize,...
                                'dEps',dEps );
            
            % report time to fit the model               
            if verbose
                stop = clock;
                fprintf( 'Training took %0.3f seconds\n',(stop(end-1)*60+stop(end)) - (start(end-1)*60+start(end)) );
            end
        end
        
        
        function get_best_clusters( self )
            % get_best_clusters( self )
            %
            % produces the optimal flat clustering from the hierarchical
            % cluster scheme in self.model by finding the most stable 
            % clusters in a recusive way
            %
            % Outputs:
            %   self.bestClusters
            %   self.corePoints
            %   self.coreLambda
            
            % check if model has been trained
            trained_check( self );
            tree = self.model.clusterTree;
            
            % get the optimal flat clustering
            self.bestClusters = OFCH( tree.stability,tree.parents );
            
            % find maximum lambda and core points for the best clusters
            [self.corePoints,self.coreLambda] = get_core_points( tree.parents,self.bestClusters,full( self.model.lambdaMax ) );
        end  
        
        
        function get_membership( self )
            % get_membership( self )
            %
            % finds the cluster membership of the points in self.data based
            % on the best clusters found in the hierarchy
            %
            % Outputs:
            %   self.labels
            %   self.score
            %   self.P
            
            % check if model has been trained           
            self.trained_check();
            
            % check if we've performed optimal flat clustering
            self.best_cluster_check();

            % compute the outlier scores
            tree = self.model.clusterTree;
            self.score = GLOSH( self.bestClusters,tree.parents,self.model.lastClust,self.model.lambdaNoise );
            
            % compute labels and probability of cluster membership
            [self.labels,self.P] = get_cluster_probability( self.bestClusters,full( self.model.lambdaMax ),self.coreLambda );
            
            % set labels with outliser scores > outlierThresh = 0
            self.labels( self.score > self.outlierThresh ) = 0;    
        end
        
        
        function [newLabels,newProb] = predict( self,newPoints )
            % [newLabels,newProb] = predict( self,newPoints )
            %
            % predicts cluster membership to new points given the trained
            % hierarchical cluster model 
            
            % check trained and best cluster assignment
            self.trained_check()
            self.best_cluster_check();
            
            % check sizes of data matrices
            [n,m] = size( newPoints );
            assert( m == self.nDims,'new data must have same # of columns as trainng data' );
            
            % create a kdtree object for finding nearest neighbors
            if isempty( self.kdtree )
                self.create_kdtree();
            end
            
            % compute the nearest neighbors of the new points and the
            % corresponding core distance of the new points
            [inds,D] = self.kdtree.knnsearch( newPoints,'K',self.minpts*2 );
            coreDist = D(:,self.minpts);
            
            % for each point, find the nearest mutual-reachable neighbor
            % among the querried raw-distance nearest neighbors
            newLambda = zeros( n,1 );
            newLabels = zeros( n,1,'int8' );
            allCoreDist = self.model.dCore';
            for i = 1:size( newPoints,1 )
                d = allCoreDist(inds(i,:));
                d(d < coreDist(i)) = coreDist(i);
                idx = D(i,:) > d;
                d(idx) = D(i,idx);
                [newLambda(i),nn] = min( d ); % the minimum mutual reach is the closest object
                newLabels(i) = self.labels(inds(i,nn));
            end
            newLambda = 1./newLambda;
            clear D inds
            
            % now that we have the lambda values, we can check if any of
            % the new points are outliers, by comparing their lambda values
            % with the minimum lambda value of the clusters they are
            % assigned to. Outliers should have lambda values that are
            % smaller than the minimum lambda value of any point belonging
            % to a certain cluster. 
            %
            % This relates to the largest "weight" in the
            % original minimum spanning tree that a point can have 
            % while still being associated with that cluster
            uniqueLabels = unique( newLabels(newLabels>0) )';
            maxLambda = full( self.model.lambdaMax );
            newProb = zeros( size( newLabels ) );
            lambdaCore = self.coreLambda;
            for k = uniqueLabels
                
                % find all points in this cluster
                inliers = self.labels == k;
                minLambda = min( maxLambda(inliers,k) ); % minimum lambda associated with this cluster
                
                % compare minimum lambda and check if any outliers
                thesePts = (newLabels == k);
                newLabels(thesePts & newLambda < minLambda) = 0;
                
                % compare the lambda values to the max lambda of this
                % cluster (the core points) to get the probability of
                % belonging to this cluster
                newProb(thesePts) = newLambda(thesePts) ./ max( lambdaCore(k),newLambda(thesePts) );
            end
        end

        
        function plot_tree( self )
            % plot_tree( self )
            % 
            % plots the cluster hierarchy tree stored in self.model
            
            % check if trained
            trained_check( self )
            
            % create the plot and change plot style
            [~,h] = plot_cluster_tree( self.model.clusterTree );
            nclusts = length( self.model.clusterTree.clusters );
            h.MarkerSize = 4;
            h.NodeColor = repmat( [0 0 0],nclusts,1 );
            h.LineStyle = '--';
            h.EdgeColor = 'k';
            h.NodeLabel = repmat( {''},1,nclusts );
            set( gca,'tickdir','out','box','off' );

            % highlight kept clusters
            if ~isempty( self.bestClusters )
                h.NodeColor(self.bestClusters,:) = repmat( [1 0 0],length( self.bestClusters ),1 );
                h.NodeLabel(self.bestClusters) = strsplit( num2str( self.bestClusters ),' ' );
            end
        end
        
        
        function h = plot_clusters( self )
            % h = plot_clusters( self )
            %
            % plots the clusters, color-coded by the labels,
            % defaulting to the first 3 columns of self.data
            %
            % Outputs:
            %   h - handle to scatter plot
            
            if self.nDims >= 3
                h = scatter3( self.data(:,1),self.data(:,2),self.data(:,3),'.' );
            else
                h = scatter( self.data(:,1),self.data(:,2),'.' );
            end
            
            % change colors according to self.labels
            if ~isempty( self.labels )
                h.CData = self.labels;
                colormap( self.cluster_colors );
            end
            
            % change appearance
            set( h.Parent,'tickdir','out','box','off','color','k','xcolor','w','ycolor','w' );
        end

    end % public methods
    
    
    %% private methods
    methods(Access=private)
        
        function trained_check( self )
            % returns an error if self.trained is false
            
            assert( isempty( self.model ),'Must train hierarchical model first!' );
        end
        
        function best_cluster_check( self )
            % returns an error if self.bestClusters is empty
            
            assert( ~isempty( self.bestClusters ),'No optimal flat clusters found.' );
        end
        
        function colors = cluster_colors( self )

             plotColor = [ 
                 [.65, .65, .65];...   % light gray         (0)
                 [0.1, 0.74, 0.95];...  % deep sky-blue     (1)
                 [0.95, 0.88, 0.05];... % gold/yellow       (2)
                 [0.80, 0.05, 0.78];... % magenta           (3)
                 [0.3, 0.8, 0.20];...   % lime green        (4)
                 [0.95, 0.1, 0.1];...   % crimson red       (5)   
                 [0.64, 0.18, 0.93];... % blue-violet       (6)
                 [0.88, 0.56, 0];...    % orange            (7)
                 [0.4, 1.0, 0.7];...    % aquamarine        (8)
                 [0.95, 0.88, 0.7];...  % salmon-yellow     (9)
                 [0, 0.2, 1];...        % blue              (10)
                 [1, 0.41, 0.7];...     % hot pink          (11)
                 [0.5, 1, 0];...        % chartreuse        (12)
                 [0.6, 0.39, 0.8];...   % amtheyist         (13)
                 [0.82, 0.36, 0.36,];...% indian red        (14)
                 [0.53, 0.8, 0.98];...  % light sky blue    (15)
                 [0, 0.6, 0.1];...      % forest green      (16)
                 [0.65, 0.95, 0.5];...  % light green       (17)
                 [0.85, 0.6, 0.88];...  % light purple      (18)
                 [0.90, 0.7, 0.7];...   % light red         (19)
                 [0.2, 0.2, 0.6];...    % dark blue         (20)
                ];

            repeats = max( 1,round( (max( self.labels )-20)/20 ) );
            colors = [plotColor(1,:);repmat( plotColor(2:end,:),repeats,1 )];
        end
        
        function create_kdtree( self )
            % creates a kdtree object based on self.data. Used for fast
            % nearest-neighbor queries for new points
            
            self.kdtree = createns( self.data,'nsmethod','kdtree' );
        end
        
    end
end
    