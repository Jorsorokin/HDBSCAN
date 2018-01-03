function model = hdbscan_fit( X,varargin )
    % model = hdbscan_fit( X,(varargin) )
    %
    % models / clusters the rows in X using the heirarchical DBSCAN, as
    % described in the manuscript:
    %
    %       Campello et al. 2015, Hierarchical Density Estimatse for Data
    %       Clustering, Visualization, and Outlier Detection
    %
    % Inputs:
    %   X - an n x m matrix with n = observations, m = variables (dimensions)
    %
    %   (minpts) - scalar specifying the minimum number of points necessary within the
    %            neighborhood radius ball of any point X(i) before that point is considered noise 
    %            for a given eps. radius. (default = 5)
    %
    %   (minclustsize) - minimum number of points for a cluster to be considered a true cluster
    %                    (default = minpts)
    %
    %
    %   (dEps) - positive int used for determining the number of epsilon
    %            (edge weights) to loop over as:
    %
    %               weights = edgeWeights(1:dEps:end)
    %               
    %            default is to use all values of 'edgeWeight'. values of
    %            dEps > 1 result in an increase in model convergence speed
    %            (as a linear scaling) at the expense of rougher
    %            cluster splitting / assignment
    % 
    %   (maxClustNum) - scalar indicating the maximum number of clusters allowed before breaking 
    %                   the search. Default = 1000
    %
    %   (outlierThresh) - a cutoff value between [0,1], where any X(i) with a GLOSH score 
    %                     (see below) greather than 'outlierThresh' is assigned
    %                     as an outlier (ID = 0). Default = 0.9
    %          
    % Outputs:
    %   model - structure produced by the clustering algorithm containing the following fields:
    %       labels      :   a 1 x n vector of final labels assigned based on the optimal "flat" clustering 
    %   
    %       labelTree   :   a matrix containing the cluster assignment (j) of each point (i) as eps is decreased.
    %                       For memory purposes, only the point at which the first cluster change occurs for any 
    %                       point and each cluster is kept
    %
    %       bestIter    :   the index of the iteration in the hierarchy
    %                       that produces a label vector most similar to
    %                       the final 'labels' vector. Useful for
    %                       reproducing the best minimum spanning tree or
    %                       for finding the epsilon value for a
    %                       non-hierarchical future DBSCAN run
    %       
    %       stability   :   matrix of cluster stabilities, where stability is measured as:
    %
    %                           S(j) = 1 - { eps_max(xi) / eps(xi) }
    %
    %                           for each cluster j
    %
    %       eps         :   vector of epsilon radius values (each value corresponds to a column 
    %                       in the hierarchy tree)  
    %   
    %       dCore       :   core distances of each X(i) as the distance to
    %                       the 'minpts' nearest neighbor of X(i)
    %
    %       corePoints  :   the core points for all clusters concatenated into a vector. 
    %                       Core points for any cluster j are the densest
    %                       points of cluster j (where P(i) >= 0.98). These
    %                       are the most "representative" points
    %
    %       coreLabels  :   the cluster labels associated with the core points
    %
    %       coreInds    :   the actual indices of the core points, for
    %                       extracting the core distances, etc.
    %      
    %       P           :   a vector of probabilities of each point X(i)
    %                       belonging to its final cluster  
    %
    %       score       :   the Global-Local Outlier Scores for each X(i),
    %                       ranging between [0,1]. Larger score values
    %                       indicate more "outlierness" 
    %
    % Written by Jordan Sorokin
    % 10/5/2017


    %% GLOBALS
    [n,m] = size( X );
    
    p = check_inputs( varargin );
    minclustsize = p.minclustsize;     
    minpts = p.minpts;
    dEps = p.dEps;
    maxClustNum = p.maxClustNum;
    outlierThresh = p.outlierThresh;
    clear p d

    %% CREATE CONENCTED TREE
    % (a) compute the core distances & mutual reachability for each X(i)
    [dCore,D] = compute_core_distances( X,minpts );
    if isscalar( D )
        mr = mutual_reachability( X,dCore );
    else
        mr = mutual_reachability( D,dCore );
        clear D
    end
    
    % (b) create the minimum spanning tree and add self loops
    method = 'dense'; type = 'tree';
    mst = minspantree( graph( mr ),'Method',method,'Type',type ); 
    mst.addedge( 1:n,1:n,dCore );

    % (c) get sorted weight vector for the loop
    epsilon = sort( unique( mst.Edges.Weight ),'descend' ); % sorted edge weights
    epsilon = epsilon(1:dEps:end);
    nEpsilon = numel( epsilon );

    % (d) pre-allocate our matrices for storage    
    isNoise = false( n,1 );
    lambda = zeros( n,2 );                  % keeps track of C(j) and epsilon when X(i) becomes noise
    parentClust = zeros( 1,maxClustNum );   % keeps track of which parent spawned each cluster
    newClusts = false( 1,maxClustNum);      % keeps track of which clusters emerge from splits
    lambdaMin = zeros( 1,maxClustNum );     % keeps track of max Eps for clust C(j) to appear
    lambdaMin(1) = 1./epsilon(1);
    lambdaMax = zeros( n,maxClustNum );     % keeps track of min Eps for point X(i) still in cluster C(j)
    currentMaxID = 1;                       % keeps track of max ID for updating labels of new components
    ID = zeros( n,nEpsilon,'int8' );        % full hierarchical label tree
    ID(:,1) = 1;
%     clustSize = zeros( 1,maxClustNum );     % keeps track of cluster size. Speeds up loop
%     clustSize(1) = n;
    
    %% HIERARCHICAL SEARCH
    for i = 2:nEpsilon
        
        newClusts(:) = false;
        ID(:,i) = ID(:,i-1); 
        
        % (e) find edges greater than current epsilon value
        idx = find( mst.Edges.Weight > epsilon(i) );
        if ~any( idx )
            continue
        end
        
        % (f) find the nodes of the cut edges and 
        % remove bad ones (those previously labeled as noise)
        endNodes = mst.Edges.EndNodes(idx,:);   % matrix of node ends for each edge
        mst = mst.rmedge( endNodes(:,1),endNodes(:,2) );
        
        % remove noise
        selfCut = (endNodes(:,1) == endNodes(:,2));
        ID(endNodes(selfCut,1)) = 0;
        endNodes(selfCut,:) = [];
        if isempty( endNodes )
            continue
        end
        
        % (g) remove noisy nodes and skip loop if no remaining nodes 
        uniqueCuts = unique( endNodes );
        badIDX = (ID(uniqueCuts,i) == 0);     
        if any( badIDX )
            if all( badIDX )
                continue
            end
            
            % if only some nodes noisy, remove those
            badNodes = uniqueCuts(badIDX);
            endNodes(any( ismember( endNodes',badNodes ) )',:) = [];
        end        

        nCutEdges = size( endNodes,1 );
        
        %% FIND SUB-TREES IN FOREST    
        for k = 1:nCutEdges
            
            % ============================
            % check if one of these nodes was already considered noise
            % from a previous loop
            parent = ID(endNodes(k,1),i-1);
                
            % get the connected components from the end nodes
            subTree = bfsearch( mst,endNodes(k,1) );
            subTree2 = bfsearch( mst,endNodes(k,2) );
            nTree1 = length( subTree );
            nTree2 = length( subTree2 );
            validTree = [nTree1,nTree2] >= minclustsize;
            
            % check for noise or splits
            % both vaild
            if all( validTree )                    
                newMax = currentMaxID + 2;
                ID(subTree,i) = newMax - 1;
                ID(subTree2,i) = newMax; 
                newClusts(newMax-1:newMax) = true;
                parentClust(newMax-1:newMax) = parent;
                currentMaxID = newMax;
                %clustSize(newMax-1) = nTree1;
                %clustSize(newMax) = nTree2;
                
            % second is noise
            elseif validTree(1)   
                ID(subTree2,i) = 0;
                %clustSize(parent) = clustSize(parent)-nTree2;
            
            % first is noise    
            elseif validTree(2)    
                ID(subTree,i) = 0;
                %clustSiz,e(parent) = nTree2;
            
            % both noise
            else                                    
                ID([subTree;subTree2],i) = 0;  
                %clustSize(parent) = 0;
            end
            % ============================
        end

        % (k) update the lambdaMin & lambdaMax variables
        changedIDs = (ID(:,i) ~= ID(:,i-1));                % find which points have changed
        if any( changedIDs )
            changeIDX = sub2ind( [n,maxClustNum],...
                find( changedIDs ),ID(changedIDs,i-1) );    % we use i-1 for columns to indicate PREVIOUS cluster
            lambdaMax(changeIDX) = 1./epsilon(i-1);         % smallest epsilon at which these points belonged to old cluster
            lambdaMin(newClusts) = 1./epsilon(i);           % largest epsion at which new clusters emerge
                
            % mark the epsilon value and last cluster C(j)
            % before which any point X(i) became noise
            isNoise(:) = (changedIDs & ID(:,i) == 0);
            lambda(isNoise,1) = ID(isNoise,i-1); % last cluster before becoming noise
            lambda(isNoise,2) = 1./epsilon(i-1); % epsilon associated with last cluster before noise
        end
        
        % break the loop from this point forward if no true clusters or too
        % many clusters 
        if ~any( ID(:,i) ) || (currentMaxID > maxClustNum)
            break 
        end
    end
    
    %% COMPUTE CLUSTER STABILITY & OPTIMAL FLAT CLUSTERING    
    lastClust = find( lambdaMin > 0,1,'last' );
    lambdaMin = lambdaMin(1:lastClust);
    lambdaMax = lambdaMax(:,1:lastClust);
    parentClust = parentClust(1:lastClust);
    inClust = ones( n,lastClust );
    inClust(lambdaMax == 0) = 0; % book keeping matrix
    S = sum( lambdaMax - bsxfun( @times,lambdaMin,inClust ) ); % [EQ 3] ... cluster stability
    clear inClust
    
    % (l) find the optimal flat clustering by traversing bottom-up
    % first to find stable clusters, then top-down to find
    % shallowest stable node of any branch. Additionally compute the 
    % outlier scores for each X(i) during the top-down traversal
        
    % Bottom-Up for optimal cluster scheme
    [delta,S] = OFCH( S,parentClust ); 

    % Top-Down for outlier scores
    score = GLOSH( delta,parentClust,lambda(:,1),lambda(:,2) );
    
    % now extract the best clusters
    S = S(delta); % [EQ 4]
    lambdaMax = lambdaMax(:,delta);
    parentClust = parentClust(delta);

    %% FINAL MODEL OUTPUT
    % compute probabilities: P( X(i) in C(j) | C(j) ) 
    % and final labels, taking outliers into account
    denom = max( lambdaMax );
    P = single( bsxfun( @rdivide,lambdaMax,denom+eps ) );  % divides by densest X(i) in each C(j)
    [~,labels] = max( P,[],2 );
    P = nansum( P,2 );
    labels = int8( labels );
    bad = (score > outlierThresh | sum( lambdaMax,2 )==0);
    labels(bad) = 0; P(bad) = 0;
    uniqueLabels = unique( labels(labels>0) );
    delta = delta( uniqueLabels );
    parentClust = parentClust( uniqueLabels );
    
    % get the "most representative" points for each cluster
%     corePoints = [];
%     coreLabels = [];
%     coreInds = [];
%     for k = keptClusts
%         coreIDX = find( labels == k );
%         [~,sortedCoreIDX] = sort( dCore(coreIDX),'ascend' );
%         corePoints = [corePoints;X(coreIDX(sortedCoreIDX(1:5)),:)];
%         coreLabels = [coreLabels,repmat(k,5,1)];
%         coreInds = [coreInds,coreIDX(sortedCoreIDX(1:5))];
%     end
    coreIDX = (score == 0 & P ~= 0);
    corePoints = X(coreIDX,:);
    coreLabels = labels(coreIDX);
    coreInds = find( coreIDX );
    
    % get the best minimum spanning tree by first finding the epsilon 
    % value associated with the most similar label vector to the final 
    % label output, then removing edges from the full msTree with weights
    % greater than this optimal epsilon
    searchRange = (epsilon < 1./max( lambdaMin(delta) ) & epsilon > 1./min( denom ));
    [~,bestIter] = min( sum( abs( ID(:,searchRange) - labels ) ) );
    bestIter = bestIter + find( searchRange,1 );
    
    % output the final model
    model = struct( 'labelTree',ID(:,1:i),'labels',labels,'eps',epsilon(1:i),...
                    'stability',S,'P',P,'score',score,'bestIter',bestIter,'dCore',dCore',...
                    'corePoints',corePoints,'coreLabels',coreLabels,'coreInds',coreInds,...
                    'clustID',delta,'clustParents',parentClust );
    %% FUNCTIONS
    function p = check_inputs( inputs )
        names = {'minpts','minclustsize','dEps',...
            'maxClustNum','computeFlatCluster','outlierThresh'};
        defaults = {5,nan,1,1000,true,0.9};

        p = inputParser;
        for j = 1:numel( names )
            p.addParameter( names{j},defaults{j} );
        end

        parse( p,inputs{:} );
        p = p.Results;

        % change minclustsize if nan
        if isnan( p.minclustsize )
            p.minclustsize = p.minpts;
        end
        
        % check dEps / round to nearest int
        p.dEps = round( p.dEps ) * sign( p.dEps );
        if p.dEps == 0
            p.dEps = 1;
        end
        
        % check outlier threshold
        if (p.outlierThresh < 0) || (p.outlierThresh > 1)
            p.outlierThresh = 0.9;
        end
    end
end