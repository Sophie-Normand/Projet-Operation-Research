"""
Représente un regroupement de données
"""
mutable struct Cluster

    dataIds::Vector{Int}
    lBounds::Vector{Float64}
    uBounds::Vector{Float64}
    x::Matrix{Float64}
    class::Any

    function Cluster()
        return new()
    end
end 

"""
Constructeur d'un cluster

Entrées :
- id : identifiant du premier élément du cluster
- x  : caractéristique des données d'entraînement
- y  : classe des données d'entraînement
"""
function Cluster(id::Int, x::Matrix{Float64}, y)

    c = Cluster()
    c.x = x[Vector{Int}([id]), :] # Crée une matrice contenant une ligne
    c.class = y[id]
    c.dataIds = Vector{Int}([id])
    c.lBounds = Vector{Float64}(x[id, :])
    c.uBounds = Vector{Float64}(x[id, :])

    return c
    
end 
