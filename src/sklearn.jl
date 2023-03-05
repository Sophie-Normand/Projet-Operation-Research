#import ScikitLearn: CrossValidation

# Set-Up related files and Hyper-parameters
#using Pkg; for p in ["ScikitLearn", "RDatasets", "DataFrames", "MLJBase"]; 
            #haskey(Pkg.installed(),p) || Pkg.add(p); end
#using ScikitLearn
#using RDatasets
#using DataFrames

#@sk_import linear_model : LogisticRegression
#@sk_import tree : DecisionTreeClassifier
#@sk_import ensemble : RandomForestClassifier
#@sk_import model_selection : train_test_split
#@sk_import metrics : accuracy_score

include("utilities.jl")

iris = dataset("datasets", "iris");

for dataSetName in ["iris", "seeds", "wine"]
        
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
    #classes = unique(Y)

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
        using Lathe.models: RandomForestClassifier
        #using Lathe.lstats: catacc

        model = RandomForestClassifier()
        model = RandomForestClassifier(max_depth = D, min_samples_leaf=1)
        model.fit(X_train, Y_train)
        yhat = model.predict(X_test)

        acc = accuracy_score(yhat,Y_test)  
    
        println("acc", acc)

        ## 2 - Multivarié
        print("    Multivarié...\t")
        model = RandomForestClassifier()
        model = RandomForestClassifier(max_depth = D)
        model.fit(X_train, Y_train)
        yhat = model.predict(X_test)

        acc = accuracy_score(yhat,Y_test)  
    
        println("acc", acc)g

    end
end

#head(iris, 10)

#features = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"];
#target = "Species";

#X = convert(Array, X);
#y = convert(Array, Y);

#struct ModelSelector
#    model_type
#    test_size
#    state
#    model_obj
#end

#function ModelSelector(test_size::Float64, 
#    state::Int; 
#    model_type="lr")

#    if model_type == "lr"
#        model_obj = LogisticRegression();
#        println("Logistic Regression Model...")
#    elseif model_type == "dt"
#        model_obj = DecisionTreeClassifier();
#        println("Decision Tree Model...")
#    elseif model_type == "rf"
#        model_obj = RandomForestClassifier();
#        println("Random Forest Model...")
#    else
#        println("Not Supported...")
#        return
#    end

#    ModelSelector(model_type, test_size, state, model_obj)

#end


#function (m::ModelSelector)(X, y)

#    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
 #                                                       test_size=m.test_size, random_state=m.state);

#    ScikitLearn.fit!(m.model_obj, X_train, Y_train);
#    preds = ScikitLearn.predict(m.model_obj, X_test);

#    acc = accuracy_score(preds, Y_test);

#    println(size(X_train), size(Y_train), size(X_test), size(Y_test))
#    println("Model Accuracy : ", acc)

#    return acc

#end


#using Lathe.models: RandomForestClassifier
#using Lathe.lstats: catacc

#model = RandomForestClassifier(X_train, Y_train)
#model = RandomForestClassifier(X_train, Y_train, n_trees = 100, max_depth = 11)
#yhat = model.predict(X_test)

#acc = accuracy_score(yhat,Y_test)

#catacc(yhat, testy)