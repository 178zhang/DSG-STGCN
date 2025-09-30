function [G, G_weighted, G_shortest_p] = get_source_neighbor_matrix(src)

    rr = cat(1, src(1).rr(logical(src(1).inuse),:), src(2).rr(logical(src(2).inuse),:));
    
    % Load and reshape triangulation meshes
    lh_use_tris = double(src(1).use_tris);
    rh_use_tris = double(src(2).use_tris);
    src(1).vertno = double(src(1).vertno);
    src(2).vertno = double(src(2).vertno);
    lh_tris = double(reshape(lh_use_tris', 1, numel(lh_use_tris)));
    rh_tris = double(reshape(rh_use_tris', 1, numel(rh_use_tris)));
    cols = size(lh_use_tris, 2);
    
    puse = length(src(1).pinfo);
    nsource = puse + length(src(2).pinfo);
    G = zeros(nsource);
    G_weighted = zeros(nsource);
    for ii = 1:puse
        % Find triangles containing iith dipole
        lh_trisno = ceil(find(ismember(lh_tris, double(src(1).vertno(ii))))/cols);
        rh_trisno = ceil(find(ismember(rh_tris, double(src(2).vertno(ii))))/cols);
        
        % Find nearest neighbor indices with respect to number of sources (lk)
        lh_vertno = setdiff(unique(lh_use_tris(lh_trisno,:)), src(1).vertno(ii));
        [~, ~, lk] = intersect(lh_vertno, src(1).vertno);
        
        rh_vertno = setdiff(unique(rh_use_tris(rh_trisno,:)), src(2).vertno(ii));
        [~, ~, rint] = intersect(rh_vertno, src(2).vertno);
        rk = rint + puse;
        
        % Form adjecency matrix (nearest neighbors)
        G(ii,lk) = 1;
        G(ii+puse,rk) = 1;
    end
    
    for ii = 1: nsource
        for jj = ii+ 1: nsource
            
            G_weighted(ii,jj) = norm(rr(ii,:)-rr(jj,:));
            
        end
    end
    
    G_weighted = G_weighted + G_weighted';
    G_weighted = G_weighted.*G;
    
    G = sparse(G);
    G_weighted = sparse(G_weighted);
    G_shortest_p = graphallshortestpaths(G_weighted)*1000;
    
end