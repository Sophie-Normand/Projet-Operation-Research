
function build_random_forest(x::Matrix{Float64}, y::Vector{}, D::Int64; multivariate::Bool=false, time_limit::Int64 = -1)

    ## Déclaration de l'objectif
    if multivariate
        classifier = RandomForestClassifier(max_depth=D, n_subfeatures=-1)
    else
        classifier = RandomForestClassifier(max_depth=D, n_subfeatures=1)
    end

    starting_time = time()
    fit!(classifier, x, y)
    resolution_time = time() - starting_time

    score_train = score(classifier, x, y)
    
    gap = (1-score_train)*100

    return classifier, resolution_time, gap
end
