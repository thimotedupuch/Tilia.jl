function _arrays(fitted)
    standardize = fitted.cpu_graph.fitted_nodes[1]
    logistic = fitted.cpu_graph.fitted_nodes[2]
    standardize, logistic
end
