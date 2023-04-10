include("building_decision_tree.jl")
include("utilities.jl")

using ScikitLearn
using DecisionTree

function main_decision_tree()

    # Pour chaque jeu de données
    for dataSetName in ["iris", "seeds", "wine", "thoracic_surgery", "diabetes_data"]
        
        print("=== Dataset ", dataSetName)

        # Préparation des données
        include("../data/" * dataSetName * ".txt")

        # Ramener chaque caractéristique sur [0, 1]
        reducedX = Matrix{Float64}(X)
        for j in 1:size(X, 2)
            reducedX[:, j] .-= minimum(X[:, j])
            reducedX[:, j] ./= maximum(X[:, j])
        end

        train, test = train_test_indexes(length(Y))
        X_train = reducedX[train, :]
        Y_train = Y[train]
        X_test = reducedX[test, :]
        Y_test = Y[test]
        classes = unique(Y)

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes
        println("Attention : le temps est fixé à 30s pour permettre de faire des tests rapides. N'hésitez pas à l'augmenter lors du calcul des résultats finaux que vous intégrerez à votre rapport.")
        time_limit = 30

        # Pour chaque profondeur considérée
        for D in 2:4

            println("  D = ", D)

            ## 1 - Univarié (séparation sur une seule variable à la fois)
            # Création de l'arbre
            print("    Univarié...  \t")
            classifier, resolution_time, gap = build_decision_tree(X_train, Y_train, D, multivariate = false, time_limit = time_limit)


            print(round(resolution_time, digits = 3), "s\t")
            print("gap ", round(gap, digits = 1), "%\t")
            print("Erreurs train/test ", prediction_errors_random_forest(classifier, X_train, Y_train))
            print("/", prediction_errors_random_forest(classifier, X_test, Y_test), "\t")
            
            println()

            ## 2 - Multivarié
            print("    Multivarié...\t")
            classifier, resolution_time, gap = build_decision_tree(X_train, Y_train, D, multivariate = true, time_limit = time_limit)

            print(round(resolution_time, digits = 3), "s\t")
            print("gap ", round(gap, digits = 1), "%\t")
            print("Erreurs train/test ", prediction_errors_random_forest(classifier, X_train, Y_train))
            print("/", prediction_errors_random_forest(classifier, X_test, Y_test), "\t")
            
            println("\n")
            print("--- Arbre de décision :")
            println("\n")
            print_tree(classifier)
            println("\n")
        end
    end 
end
