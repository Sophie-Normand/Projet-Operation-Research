using Distances

"""
Représente une distance entre deux données identifiées par leur id
"""
mutable struct Distance

    distance::Float64
    ids::Vector{Int}

    function Distance()
        return new()
    end
end 

"""
Constructeur d'une distance

Entrées :
- id1 : id de la première donnée
- id2 : id de la seconde donnée
- x   : caractéristique des données d'entraînement
"""
function Distance(id1::Int, id2::Int, x::Matrix{Float64})

    d = Distance()
    d.distance = euclidean(x[id1, :], x[id2, :])
    d.ids = [id1, id2]

    return d
    
end
