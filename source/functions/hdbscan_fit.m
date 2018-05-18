function model = hdbscan_fit( X,varargin )
    % model = hdbscan_fit( X,(varargin) )
    %
    % models the rows in X using the heirarchical DBSCAN, as
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
    %   (minClustNum) - the minimum # of clusters; the first minClustNum
    %                   parent clusters will have stabilities set to 0
    %                   (default = 1)
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
    % Outputs:
    %   model - structure produced by the clustering algorithm containing the following fields:
    %
    %       clusterTree :   a structure with the cluster tree created by the data in X. 
    %                       The tree is condensed in that spurious components (where
    %                       # pts < minclustsize) are eliminated. The
    %                       resulting tree is orders of magnitude smaller
    %                       than the full hierarchy. 
    %
    %                       The structure contains:
    %                           clusters - vector of cluster labels
    %                           parents - their parents
    %                           lambdaMin - minimum lambda (max epsilon) at which each was created
    %                           stability - their stabilities
    %                       
    %                       stabilities are measured as:
    %
    %                           S(j) = 1 - { eps_max(xi) / eps(xi) }
    %
    %       lambda      :   vector of 1 / epsilon radii
    %
    %       lambdaNoise :   the lambda associated with point i beyond which that
    %                       point becomes noise (ID = 0). This is the
    %                       smallest radii (largest lambda) that point i belongs 
    %                       to any cluster j
    %
    %       lambdaMax   :   the lambda associated with point i and cluster
    %                       j after which point i no longer belongs to
    %                       cluster j ... (1 / eps(xi) )
    %
    %       lastClust   :   the final cluster j that point i belonged to
    %                       before becoming noise
    %   
    %       dCore       :   core distances of each X(i) as the distance to
    %                       the 'minpts' nearest neighbor of X(i)
    %
    % Written by Jordan Sorokin
    % 10/5/2017


    %% GLOBALS
    [n,~] = size( X );    
    p = check_inputs( varargin );
    minclustsize = p.minclustsize;     
    minpts = p.minpts;
    dEps = p.dEps;
    maxClustNum = 1000;
    minClustNum = p.minClustNum;
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
    nodes = mst.Edges.EndNodes;
    weights = mst.Edges.Weight;
    
    % (c) get sorted weight vector for the loop
    epsilon = sort( unique( weights ),'descend' ); % sorted edge weights
    epsilon = epsilon(1:dEps:end);
    nEpsilon = numel( epsilon );

    % (d) pre-allocate our matrices for storage  
    lambdaNoise = zeros( n,1 );             % keeps track of epsilon when X(i) becomes noise
    lastClust = zeros( n,1 );               % keepst rack of C(j) when X(i) becomes noise
    parentClust = zeros( 1,maxClustNum );   % keeps track of which parent spawned each cluster
    lambdaMin = zeros( 1,maxClustNum );     % keeps track of max Eps for clust C(j) to appear
    lambdaMax = zeros( n,maxClustNum );     % keeps track of min Eps for point X(i) still in cluster C(j)
    currentMaxID = 1;                       % keeps track of max ID for updating labels of new components
    lambdaMin(1) = 1./epsilon(1);
    newID = ones( n,1,'uint8' );       
    
    %% HIERARCHICAL SEARCH
    for i = 2:nEpsilon
        
        oldID = newID;
        
        % (e) find edges greater than current epsilon value
        idx = weights > epsilon(i);
        if ~any( idx )
            continue
        end
        
        % (f) find the nodes of the cut edges and 
        % remove bad ones (those previously labeled as noise)
        endNodes = nodes(idx,:);
        nodes(idx,:) = [];
        weights(idx) = []; 
        mst = mst.rmedge( endNodes(:,1),endNodes(:,2) );

        % remove noise
        selfCut = (endNodes(:,1) == endNodes(:,2));
        if any( selfCut )
            newID(endNodes(selfCut,1)) = 0;
            endNodes(selfCut,:) = [];
            if isempty( endNodes )
                continue
            end
        end
        
        % (g) remove noisy nodes and skip loop if no remaining nodes 
        uniqueCuts = unique( endNodes );
        badIDX = (newID(uniqueCuts) == 0);     
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
            
            % (h) get the connected components from the end nodes
            parent = oldID(endNodes(k,1));
            
            subTree = bfsearch( mst,endNodes(k,1) );
            subTree2 = bfsearch( mst,endNodes(k,2) );
            nTree1 = length( subTree );
            nTree2 = length( subTree2 );
            validTree = [nTree1,nTree2] >= minclustsize;            
            
            % (i) check for noise or splits
            % (i.1) - both vaild
            if validTree
                newMax = currentMaxID + 2;
                temp = newMax - 1;
                newID(subTree) = temp;
                newID(subTree2) = newMax; 
                parentClust(temp:newMax) = parent;
                currentMaxID = newMax;
                lambdaMax([subTree;subTree2],parent) = epsilon(i-1);
                lambdaMin([temp,newMax]) = epsilon(i);
            else
                % (i.2) - second is noise
                if validTree(1)   
                    isNoise = subTree2;
                elseif validTree(2)
                    isNoise = subTree;
                else
                    isNoise = [subTree;subTree2];
                end
                
                lastClust(isNoise) = oldID(isNoise);
                lambdaNoise(isNoise) = epsilon(i-1);
                lambdaMax(isNoise,parent) = epsilon(i-1);
                newID(isNoise) = 0;
                
                if ~any( newID )
                    break
                end    
            end
        end
    end
    
    %% COMPUTE CLUSTER STABILITY % OUTPUT MODEL 
    
    % (j) get the condensed tree
    lambdaMin = 1 ./ lambdaMin(1:currentMaxID);
    lambdaMax = 1 ./ lambdaMax(:,1:currentMaxID);
    lambdaMax(isinf(lambdaMax)) = 0;
    parentClust = parentClust(1:currentMaxID);
    
    % (k) compute stabilities
    inClust = ones( n,currentMaxID );
    inClust(lambdaMax == 0) = 0; 
    S = sum( lambdaMax - bsxfun( @times,lambdaMin,inClust ) ); % [EQ 3] ... cluster stability    
    
    % set 1:minClustNum-1 parent clust stabilities to 0
    if minClustNum > 1
        [uniqueParents,parentLocation] = unique( parentClust(parentClust>0) );
        [~,idx] = sort( parentLocation );
        uniqueParents = uniqueParents( idx ); % order in which the parents were split
        nchildren = zeros( 1,numel( uniqueParents ) );
        for i = 1:numel( uniqueParents )
            nchildren(i) = nnz( parentClust == uniqueParents(i) );
        end
        
        % set stabilities for the first stability-ordered N parents that, cumulatively, have
        % minClustNum # of children equal to 0 
        removeParents = uniqueParents( 1:find( cumsum( nchildren ) >= minClustNum,1 ) );
        S(removeParents) = 0;
    end
    
    % Model output
    clusterTree = struct( 'clusters',1:currentMaxID,'parents',parentClust,...
        'lambdaMin',lambdaMin,'stability',S ); 
    
    model = struct( 'lambda',1./epsilon(1:i),'clusterTree',clusterTree,'dCore',dCore,...
                    'lambdaNoise',1./lambdaNoise,'lastClust',lastClust,'lambdaMax',sparse( lambdaMax ) );

    %% FUNCTIONS
    function p = check_inputs( inputs )
        names = {'minpts','minclustsize','minClustNum','dEps'};
        defaults = {5,nan,1,1};

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
    end
end
