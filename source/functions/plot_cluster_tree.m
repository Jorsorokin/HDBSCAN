function [G,h] = plot_cluster_tree( tree )
    % [G,h] = plot_cluster_tree( tree )
    %
    % plots the condensed cluster tree as a connected 
    % graph, based on the relationship between parent clusters
    % and all sub clusters from the parent
    %
    % Inputs:
    %   tree - a cluster tree created by HDBSCAN.fit() or hdbscan_fit
    %
    % Outputs:
    %   G - the graph object created by linking a parent with all 
    %       sub clusters spawning from the parent
    %
    %   h - the handle to the figure
    %
    % Written by Jordan Sorokin, 10/15/2017

    % check inputs
    clusters = tree.clusters;
    parents = tree.parents;
    weights = tree.lambdaMin;
    if parents(1) == 0
        parents = parents(2:end);
        clusters = clusters(2:end);
        weights = [0,weights(2:end)];
    end
   
    % create a graph object and plot
    G = graph( clusters,parents );
    h = G.plot();
    h.YData = weights;
    h.Parent.YDir = 'reverse';
    h.Parent.YLabel.String = '\lambda';
end
    