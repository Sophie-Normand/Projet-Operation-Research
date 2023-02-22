include("struct/distance.jl")

"""
Essaie de regrouper des données en commençant par celles qui sont les plus proches.
Deux clusters de données peuvent être fusionnés en un cluster C s'il n'existe aucune données x_i pour aucune caractéristique j qui intersecte l'intervalle représenté par les bornes minimale et maximale de C pour j (x_i,j n'appartient pas à [min_{x_k dans C} x_k,j ; max_{k dans C} x_k,j]).

Entrées :
- x : caractéristiques des données d'entraînement
- y : classe des données d'entraînement
- percentage : le nombre de clusters obtenu sera égal à n * percentage
 
Sorties :
- un tableau de Cluster constituant une partition de x
"""
function exactMerge(x, y)

    n = length(y)
    m = length(x[1,:])
    
    # Liste de tous les clusters obtenus
    # (les données non regroupées seront seules dans un cluster)
    clusters = Vector{Cluster}([])

    # Initialement, chaque donnée est dans un cluster
    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(dataId, x, y))
    end

    # Id du cluster de chaque donnée dans clusters
    # (clusters[clusterId[i]] est le cluster contenant i)
    # (clusterId[i] = i, initialement)
    clusterId = collect(1:n)

    # Distances entre des couples de données de même classe
    distances = Vector{Distance}([])

    # Pour chaque couple de données de même classe
    for id1 in 1:n-1
        for id2 in id1+1:n
            if y[id1] == y[id2]
                # Ajoute leur distance
                push!(distances, Distance(id1, id2, x))
            end
        end
    end

    # Trie des distances par ordre croissant
    sort!(distances, by = v -> v.distance)
    
    # Pour chaque distance
    for distance in distances

        # Si les deux données associées ne sont pas déjà dans le même cluster
        cId1 = clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]

        if cId1 != cId2
            c1 = clusters[cId1]
            c2 = clusters[cId2]

            # Si leurs clusters satisfont les conditions de fusion
            if canMerge(c1, c2, x, y)

                # Les fusionner
                merge!(c1, c2)
                for id in c2.dataIds
                    clusterId[id]= cId1
                end

                # Vider le second cluster
                empty!(clusters[cId2].dataIds)
            end 
        end 
    end

    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end

"""
Regroupe des données en commençant par celles qui sont les plus proches jusqu'à ce qu'un certain pourcentage de clusters soit atteint

Entrées :
- x : caractéristiques des données
- y : classe des données
- gamma : le regroupement se termine quand il reste un nombre de clusters < n * gamma ou que plus aucun regroupement n'est possible

Sorties :
- un tableau de Cluster constituant une partition de x
"""
function simpleMerge(x, y, gamma)

    n = length(y)
    m = length(x[1,:])
    
    # Liste de tous les clusters obtenus
    # (les données non regroupées seront seules dans un cluster)
    clusters = Vector{Cluster}([])

    # Initialement, chaque donnée est dans un cluster
    for dataId in 1:size(x, 1)
        push!(clusters, Cluster(dataId, x, y))
    end

    # Id du cluster de chaque donnée dans clusters
    # (clusters[clusterId[i]] est le cluster contenant i)
    # (clusterId[i] = i, initialement)
    clusterId = collect(1:n)

    # Distances entre des couples de données de même classe
    distances = Vector{Distance}([])

    # Pour chaque couple de données de même classe
    for id1 in 1:n-1
        for id2 in id1+1:n
            if y[id1] == y[id2]
                # Ajoute leur distance
                push!(distances, Distance(id1, id2, x))
            end
        end
    end

    # Trie des distances par ordre croissant
    sort!(distances, by = v -> v.distance)

    remainingClusters = n
    distanceId = 1

    # Pour chaque distance et tant que le nombre de cluster souhaité n'est pas atteint
    while distanceId <= length(distances) && remainingClusters > n * gamma

        distance = distances[distanceId]
        cId1 = clusterId[distance.ids[1]]
        cId2 = clusterId[distance.ids[2]]

        # Si les deux données associées ne sont pas déjà dans le même cluster
        if cId1 != cId2
            remainingClusters -= 1

            # Fusionner leurs clusters 
            c1 = clusters[cId1]
            c2 = clusters[cId2]
            merge!(c1, c2)
            for id in c2.dataIds
                clusterId[id]= cId1
            end

            # Vider le second cluster
            empty!(clusters[cId2].dataIds)
        end
        distanceId += 1
    end
    
    # Retourner tous les clusters non vides
    return filter(x -> length(x.dataIds) > 0, clusters)
end 

"""
Test si deux clusters peuvent être fusionnés tout en garantissant l'optimalité

Entrées :
- c1 : premier cluster
- c2 : second cluster
- x  : caractéristiques des données d'entraînement
- y  : classe des données d'entraînement

Sorties :
- vrai si la fusion est possible ; faux sinon.
"""
function canMerge(c1::Cluster, c2::Cluster, x::Matrix{Float64}, y::Vector{Int})

    # Calcul des bornes inférieures si c1 et c2 étaient fusionnés
    mergedLBounds = min.(c1.lBounds, c2.lBounds)
    
    # Calcul des bornes supérieures si c1 et c2 étaient fusionnés
    mergedUBounds = max.(c1.uBounds, c2.uBounds)

    n = size(x, 1)
    id = 1
    canMerge = true

    # Tant que l'ont a pas vérifié que toutes les données n'intersectent la fusion de c1 et c2 sur aucune feature
    while id <= n && canMerge

        data = x[id, :]

        # Si la donnée n'est pas dans c1 ou c2 mais intersecte la fusion de c1 et c2 sur au moins une feature
        if !(id in c1.dataIds) && !(id in c2.dataIds) && isInABound(data, mergedLBounds, mergedUBounds)
            canMerge = false
        end 
        
        id += 1
    end 

    return canMerge
end

"""
Test si une donnée intersecte des bornes pour au moins une caractéristique 

Entrées :
- v : les caractéristique de la donnée
- lowerBounds : bornes inférieures pour chaque caractéristique
- upperBounds : bornes supérieures pour chaque caractéristique

Sorties :
- vrai s'il y a intersection ; faux sinon.
"""
function isInABound(v::Vector{Float64}, lowerBounds::Vector{Float64}, upperBounds::Vector{Float64})
    isInBound = false

    featureId = 1

    # Tant que toutes les features n'ont pas été testées et qu'aucune intersection n'a été trouvée
    while !isInBound && featureId <= length(v)

        # S'il y a intersection
        if v[featureId] >= lowerBounds[featureId] && v[featureId] <= upperBounds[featureId]
            isInBound = true
        end 
        featureId += 1
    end 

    return isInBound
end

"""
Fusionne deux clusters

Entrées :
- c1 : premier cluster
- c2 : second cluster

Sorties :
- aucune, c'est le cluster en premier argument qui contiendra le second
"""
function merge!(c1::Cluster, c2::Cluster)

    append!(c1.dataIds, c2.dataIds)
    c1.x = vcat(c1.x, c2.x)
    c1.lBounds = min.(c1.lBounds, c2.lBounds)
    c1.uBounds = max.(c1.uBounds, c2.uBounds)    
end
