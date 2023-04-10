using Random
include("struct/tree.jl")

"""
Création de deux listes d'indices pour les jeux de données d'entrainement et de test
Entrées :\n
    - n le nombre de données dans le dataset
    - p la proportion que représente le jeu de données test
Sorties :\n
    - train, la liste des indices des données d'entrainement
    - test, la liste des indices des données de tesr
"""
function train_test_indexes(n::Int64,p::Float64=0.2)

    # Fixe la graine aléatoire pour assurer la reproductibilité
    Random.seed!(1)
    rd = randperm(n)

    test = rd[1:ceil(Int,n*p)]
    train = rd[ceil(Int,n*p)+1:n]

    return train,test
end

"""
Retourne le nombre d'erreurs de prédiction d'un arbre pour un ensemble de données

Entrées :
- T : l'arbre
- x : les données à prédire
- y : la classe des données

Sortie :
- class::Vector{Int64} : class prédites (class[i] est la classe de la donnée x[i, :])
"""
function prediction_errors(T::Tree, x::Matrix{Float64}, y::Vector{}, classes::Vector{})
    dataCount = length(x[:, 1])
    featuresCount = length(x[1, :])
    
    errors = 0

    # Pour chaque donnée i
    for i in 1:dataCount
        t = 1

        # Pour chaque profondeur de la branche suivie par i
        for d in 1:(T.D+1)

            # Si le sommet t atteint prédit une classe, l'associer à i
            if T.c[t] != -1
                errors += classes[T.c[t]] != y[i]
                break
            else # Sinon atteindre le sommet fils suivant dans la branche de i
                if sum(T.a[j, t]*x[i, j] for j in 1:featuresCount) - T.b[t] < 0
                    t = t*2
                else
                    t = t*2 + 1
                end
            end
        end
    end
    return errors
end

function prediction_errors_random_forest(classifier, x::Matrix{Float64}, y::Vector{})
    y_pred = predict(classifier, x)

    errors = 0
    for i in 1:length(y)
        if y[i] != y_pred[i]
            errors += 1
        end
    end
    return errors
end

"""
Change l'échelle des caractéristiques d'un dataset pour les situer dans [0, 1]

Entrée :
- X: les caractéristiques du dataset d'origine

Sortie :
- caractéristiques entre 0 et 1
"""
function centerData(X)

    result = Matrix{Float64}(X)

    # Pour chaque caractéristique
    for j in 1:size(result, 2)
        
        m = minimum(result[:, j])
        M = maximum(result[:, j])
        result[:, j] .-= m
        result[:, j] ./= M
    end

    return result
end

function centerAndSaveDataSet(X, Y::Vector{Int64}, outputFile::String)
    
    centeredX = centerData(X)

    open(outputFile, "w") do fout
        println(fout, "X = ", centeredX)
        println(fout, "Y = ", Y)
    end    
end 
