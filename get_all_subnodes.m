function children = get_all_subnodes( u,parents )
    % children = get_all_subnodes( u,parents )
    %
    % recursively finds all subnodes of a branching tree 
    % starting from root node "u"
    %
    % Inputs:
    %   u - the root node to search top-down from
    %
    %   parents - a list of parents of length nNode,
    %             where nNode = total # of nodes in the 
    %             tree, and parents(i) indicates the directly-
    %             connected parent to the ith node
    %
    %       i.e. for a branching tree such as:
    %
    %                       (1)
    %                      __|__
    %                    (2)   (3)
    %                   __|__  
    %                 (4)   (5)
    %
    %       the vector 'parent' would be: [0 1 1 2 2]
    %
    % Written by Jordan Sorokin, 10/11/2017
    
    % find children of this node
    subChildren = find(parents == u); 
    if isempty( subChildren )
        children = [];
        return
    end
    
    % add children to growing list
    children = subChildren;
    
    % recursively find children for each new child
    for child = subChildren
        newChildren = get_all_subnodes( child,parents );
        children = [children,newChildren];
    end
end