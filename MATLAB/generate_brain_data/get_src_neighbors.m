function src_level_neighbors = get_src_neighbors(src_graph_binary, level)

if level == 1
    src_level_neighbors = src_graph_binary;
else % level > 1
    temp = src_graph_binary;
    for i = 2:level
        temp = temp * src_graph_binary;
    end
    src_level_neighbors = temp - diag(diag(temp));
    src_level_neighbors = (src_level_neighbors>0);
end