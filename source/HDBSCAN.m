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
    %   nPoints         -   the number of rows in the matrix "data"
    %
    %   nDims           -   the number of columns in the matrix "data"
    %   
    %   minpts          -   the nearest 'minpts' neighbor used for core distance
    %                       calculation for each point in X. Default = 5
    %   
    %   minclustsize    -   the minimum # of points necessary for a cluster
    %                       to be deemed valid. Default = 5
    %
    %   minClustNum     -   the minimum # of clusters to be realized. Default = 1
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
    %   clusterMap      -   maps each unique ID in labels to the best
    %                       cluster it is associated with
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
    %   fit_model       -   fits a hierarchical model to the data in X
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
    %   plot_clusters   -   plots the first 3 (or 2, if 2D) columns of self.data,
    %                       color coded by the cluster labels of the data points
    %
    %   run_hdbscan     -   convenience function that fits a full
    %                       hierarchical model, finds optimal clusters, and
    %                       assigns labels to data points
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
        minClustNum = 1;
        outlierThresh = 0.9;
        bestClusters
        clusterMap
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
            
        
        function fit_model( self,varargin )
            % fit_model( self,(dEps,verbose) )
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
                fprintf( '\tMin # of clusters: %i\n',self.minClustNum );
                fprintf( '\tSkipping every %i iteration\n\n',dEps-1 );
                start = clock;
            end
                
            % fit the hierarchical cluster tree
            self.model = hdbscan_fit( self.data,...
                                'minpts',self.minpts,...
                                'minclustsize',self.minclustsize,...
                                'minClustNum',self.minClustNum,...
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
            self.trained_check();
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
            self.clusterMap = unique( self.labels(self.labels>0) )';

            % set labels with outlier scores > outlierThresh = 0
            self.labels( self.score > self.outlierThresh ) = 0;    
            
            % update if any clusters are now all zero
            badclusts = ~ismember( self.clusterMap,unique( self.labels(self.labels>0) ) );
            self.clusterMap( badclusts ) = [];
            self.corePoints = self.corePoints( self.clusterMap );
            self.coreLambda = self.coreLambda( self.clusterMap );
            self.bestClusters = self.bestClusters( self.clusterMap );
        end
        
        
        function run_hdbscan( self,varargin )
            % run_hdbscan( self,(minpts,minclustsize,minClustNum,dEps,outlierThresh,plotResults) )
            %
            % fits a hierarchical model to self.data and finds the best
            % flat clustering scheme. Then assigns labels to each data
            % point in self.data based on the final clusters.
            %
            %   Note: this is just a convenience function to avoid manually
            %         typing the commands to perform these operations
            %
            % Inputs:
            %   minpts - minimum # neighbors for core distances
            %
            %   minclustsize - minimum # points in a cluster to keep the
            %                  cluster
            %
            %   minClustNum - the minimum # of clusters to be discovered. Default = 1
            %
            %   dEps - iterator (skips edge weight iteratios every "dEps"
            %          times)
            %
            %   outlierThresh - threshold between [0,1] for outlier scores
            %
            %   plotResults - logical to plot the cluster results or not
            %
            % Outputs:
            %   self.model
            %   self.corePoints
            %   self.coreLabels
            %   self.bestClusters
            %   self.labels
            %   self.P
            %   self.score
            
            % check inputs
            if nargin > 1 && ~isempty( varargin{1} )
                self.minpts = varargin{1};
            end
            if nargin > 2 && ~isempty( varargin{2} )
                self.minclustsize = varargin{2};
            end
            if nargin > 3 && ~isempty( varargin{3} )
                self.minClustNum = varargin{3};
            else
                self.minClustNum = 1;
            end
            if nargin > 4 && ~isempty( varargin{4} )
                dEps = varargin{4};
            else
                dEps = 1;
            end
            if nargin > 5 && ~isempty( varargin{5} )
                self.outlierThresh = varargin{5};
            end
            if nargin > 6 && ~isempty( varargin{6} )
                plotResults = varargin{6};
            else
                plotResults = false;
            end
            
            % fit the hierarchical model
            self.fit_model( dEps );
            
            % extract best clusters
            self.get_best_clusters();
            
            % assign labels
            self.get_membership();
            
            % visualize the results
            if plotResults
                figure;
                self.plot_tree();
                
                figure;
                self.plot_clusters();
                set( gcf,'color','k' )
            end
        end
        
        
        function update_hierarchy( self,newLabels )
            % update_hierarchy( self,newLabels )
            %
            % updates the cluster hierarchy and the lambda values associated
            % with the clusters based on any new label vector. This allows
            % one to manually alter the clusters while maintaining a
            % probabilistic model that can be used to predict new points
            %
            % Inputs:
            %   newLabels - self.nPoints x 1 vector of new labels
            %
            % Outputs:
            %   updates all properties pertaining to the cluster hierarchy
            %   in "self"
            
            % check for model / best clusters
            self.trained_check();
            self.best_cluster_check();
            if isrow( newLabels )
                newLabels = newLabels';
            end
            
            % get the necessary variables that will be updated
            lambdaMax = full( self.model.lambdaMax );
            lambdaNoise = self.model.lambdaNoise;
            bestClusts = self.bestClusters;
            map = self.clusterMap;
            parents = self.model.clusterTree.parents;
            clusters = self.model.clusterTree.clusters;
            minLambda = self.model.clusterTree.lambdaMin;
            stability = self.model.clusterTree.stability;
            nClusts = clusters(end);
            newClusters = [];
            
            % find changed labels
            oldLabels = self.labels;
            changedPts = (oldLabels ~= newLabels);
            changedLabels = unique( newLabels(changedPts) )';
            changedLabels(changedLabels == 0) = [];

            % loop over the changed clusters, and update the model 
            % depending on whether the new cluster is a result 
            % of a split or merge
            for k = changedLabels
                pts = changedPts & (newLabels == k); % intersection{ C_k, C_i }
                prevID = unique( oldLabels(pts) );
                prevClust = bestClusts( ismembc( map(~ismembc(bestClusts,newClusters)),prevID ) );
                thisClust = bestClusts( map == k );
                
                switch numel( prevClust )
                    case 0 % previous points were just noise
                        
                        % update lambdas of points by using the lambdas at
                        % which they become noise
                        nClusts = nClusts + 1;
                        newLambda = lambdaNoise(pts);
                        lambdaMax(:,nClusts) = 0;
                        lambdaMax(pts,nClusts) = newLambda;
                        
                        % update cluster stability/lambda by taking the
                        % mean of the parent clusters
                        clusters(nClusts) = nClusts;
                        parents(nClusts) = max( round( mean( self.model.lastClust(pts) ) ),1 ); % this is a hack
                        minLambda(nClusts) = minLambda(parents(nClusts));
                        stability(nClusts) = stability(parents(nClusts));
                        bestClusts(end+1) = nClusts;
                        map(end+1) = k;
                        newClusters(end+1) = nClusts;
                                  
                    case 1 && ~any( thisClust ) % split / manual new cluster
                        
                        % update the lambdas of the new clusters by simply
                        % moving lambdas associated with appropraite points
                        nClusts = nClusts + 1;
                        newLambda = lambdaMax(pts,prevClust);
                        lambdaMax(pts,prevClust) = 0;
                        lambdaMax(:,nClusts) = 0;
                        lambdaMax(pts,nClusts) = newLambda;
                        
                        % update cluster minimum lambda by just copying
                        % from previous one
                        minLambda(nClusts) = minLambda(prevClust); % just take a copy
                        stability(nClusts) = stability(prevClust); % ditto
                        
                        % update the clusters and parents vectors
                        clusters(nClusts) = nClusts;
                        parents(nClusts) = prevClust;
                        bestClusts(end+1) = nClusts;
                        map(end+1) = k;
                        newClusters(end+1) = nClusts;

                    otherwise % merge
                        
                        % check if the current cluster is a new cluster,
                        % resulting from a merge that produced an entirely
                        % new ID, rather than merging with a previous ID
                        if isempty( thisClust )
                            nClusts = nClusts + 1;
                            bestClusts(end+1) = nClusts;
                            map(end+1) = k;
                            newClusters(end+1) = nClusts;
                            thisClust = nClusts;
                            lambdaMax(:,thisClust) = 0;
                            minLambda(thisClust) = 0;
                            stability(thisClust) = 0;
                        end
                        
                        for j = 1:numel( prevID )
                            
                            % update the lambdas
                            oldpts = (oldLabels == prevID(j));
                            ptFrac = nnz( oldpts ) / nnz( newLabels==k );
                            switch prevID(j) 
                                case 0 % just noise
                                    %oldCluster = mode( self.model.lastClust(oldpts) );
                                    oldLambda = lambdaNoise(oldpts);
                                    
                                otherwise % prev cluster used
                                    oldCluster = bestClusts(map == prevID(j) & ~ismember( bestClusts,newClusters ));
                                    oldLambda = lambdaMax(oldpts,oldCluster);
                                    lambdaMax(oldpts,oldCluster) = 0;
                                    
                                    % update the minLambda and stability vectors
                                    % by taking a weighted average, determined by
                                    % the fraction of points merged from cluster j
                                    minLambda(thisClust) = (minLambda(thisClust) + ptFrac*minLambda(oldCluster)) / 2;
                                    stability(thisClust) = (stability(thisClust) + ptFrac*stability(oldCluster)) / 2;
                            end
                            
                            lambdaMax(oldpts,thisClust) = oldLambda;
                        end 
                end
            end
                        
            % eliminate old clusters that are now just noise or eliminated
            newClusts = unique( newLabels(newLabels > 0) );
            badClusts = ~ismembc( map,newClusts );
            bestClusts(badClusts) = [];
            map(badClusts) = [];
            
            % now find the core points and core lambda for the clusters
            [self.corePoints,self.coreLambda] = get_core_points( parents,bestClusts,lambdaMax );

            % store the updated parameters
            for i = 1:numel( map )
                newLabels(newLabels==map(i)) = i;
                map(i) = i;
            end
            self.bestClusters = bestClusts;
            self.clusterMap = map;
            self.model.lambdaMax = sparse( lambdaMax );
            self.model.clusterTree.clusters = clusters;
            self.model.clusterTree.parents = parents;
            self.model.clusterTree.stability = stability;
            self.model.clusterTree.lambdaMin = minLambda;
            
            % update the labels
            self.labels = newLabels;
        end
        
        
        function [newLabels,newProb,outliers] = predict( self,newPoints )
            % [newLabels,newProb,outliers] = predict( self,newPoints,alpha )
            %
            % predicts cluster membership to new points given the trained
            % hierarchical cluster model.
            %
            % For each point i, prediction is performed as follows:
            %   (a) set D(i,j) = euclidean distance of the jth nearest
            %       neighbor with label > 0, for j in [1, self.minpts*2]
            %
            %   (b) set R(i) = nearest self.minpts mutual-reachability 
            %       distance among the self.minpts*2 nearest neighbors.
            %       
            %   (c) assign label(i) = label of the nearest mutual-reachable
            %       neighbor of point i
            %   
            %   (d) set L(i) = lambda value for point i as 1 / R(i)
            %
            %   (e) P(i) = L(i) / L_max; L_max = maximum lambda of the
            %       cluster assigned to point i
            %
            %   (f) flag outlier(i) IF 1-P(i) > self.outlierThresh
            
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

            % for each point, find the nearest core points 
            uID = unique( self.labels(self.labels>0) );
            nID = numel( uID );
            D = zeros( n,nID ); 
            for i = 1:nID
                points = self.data( self.corePoints{i},: );
                d = compute_pairwise_dist( newPoints,points );
                D(:,i) = min( d,[],2 );
            end
            
            % convert the mutual reaches to lambda values
            [newLambda,newLabels] = min( D,[],2 );
            newLambda = 1./newLambda;
            
            % now that we have the lambda values, we can check if any of
            % the new points are outliers, by comparing their lambda values
            % with the minimum lambda value of the clusters they are
            % assigned to. This relates to the largest "weight" in the
            % original hierarchical tree that a point can have 
            % while still being associated with its particular cluster
            uniqueLabels = unique( newLabels(newLabels>0) )';
            newProb = zeros( size( newLabels ) );
            lambdaCore = self.coreLambda;
            map = self.clusterMap;

            % compare the lambda values to the max lambda of this
            % cluster (the core points) to get the probability of
            % belonging to this cluster
            for k = uniqueLabels
                thesePts = (newLabels == k);
                newProb(thesePts) = newLambda(thesePts) ./ max( lambdaCore(map == k),newLambda(thesePts) );
            end
            
            % outlier if 1 - probability is > outlier threshold
            outliers = find( (1-newProb) > self.outlierThresh );
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
            set( gca,'tickdir','out','box','off','XTick',[],'XTickLabel',[] );
            title( 'Condensed cluster tree' );

            % highlight kept clusters
            if ~isempty( self.bestClusters )
                h.NodeColor(self.bestClusters,:) = repmat( [1 0 0],length( self.bestClusters ),1 );
                h.NodeLabel(self.bestClusters) = strsplit( num2str( self.bestClusters ),' ' );
            end
        end
        
        
        function h = plot_clusters( self,varargin )
            % h = plot_clusters( self,(dims) )
            %
            % plots the clusters, color-coded by the labels,
            % defaulting to the first 3 columns of self.data
            %
            % Inputs:
            %   (dims) - up to 3 dimensions (columns) of self.data to plot.
            %            Must specify self.nDims different dims to plot
            %
            % Outputs:
            %   h - handle to scatter plot
            
            if nargin > 1 && ~isempty( varargin{1} )
                dims = varargin{1};
                dims = dims(1:min( self.nDims,3 ));
            else
                dims = 1:self.nDims;
            end
            
            % scatter plots
            if self.nDims >= 3
                h = scatter3( self.data(:,dims(1)),self.data(:,dims(2)),self.data(:,dims(3)),'.' );
            else
                h = scatter( self.data(:,dims(1)),self.data(:,dims(2)),'.' );
            end
            
            % change colors according to self.labels
            if ~isempty( self.labels )
                h.CData = self.labels;
                colormap( self.cluster_colors );
            end
            
            % change appearance
            title( 'Clustered data','color','w' );
            set( h.Parent,'tickdir','out','box','off','color','k','xcolor','w','ycolor','w' );
        end

    end % public methods
    
    
    %% private methods
    methods(Access=private)
        
        function trained_check( self )
            % returns an error if self.trained is false
            
            assert( ~isempty( self.model ),'Must train hierarchical model first!' );
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

            repeats = max( 1,ceil( max( self.labels )/19 ) );
            colors = [plotColor(1,:);repmat( plotColor(2:end,:),repeats,1 )];
        end
        
        function create_kdtree( self )
            % creates a kdtree object based on self.data. Used for fast
            % nearest-neighbor queries for new points
            
            self.kdtree = createns( self.data(self.labels>0,:),'nsmethod','kdtree' );
        end
        
    end
end
    
