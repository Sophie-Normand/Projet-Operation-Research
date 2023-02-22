using JuMP
using CPLEX

include("cluster.jl")

"""
Structure représentant un arbre de décision
"""
mutable struct Tree

    D::Int64 # Nombre maximal de séparations d'une branche (profondeur de l'arbre - 1)
    a::Matrix{Float64} # Vecteurs caractérisant les règles de branchement (a.x < b)
    b::Vector{Float64} # Idem
    c::Vector{Int64} # c[t] : classe du noeud d'indice t (-1 si aucune classe n'est attribuée)
    
    function Tree()
        return new()
    end
end

"""
Constructeur par défaut d'un arbre
(les séparations sont celles fournies par le solveur, sans recentrage ni post-traitement)

Entrée :
- D : Nombre maximal de séparations d'une branche (profondeur de l'arbre - 1)
- a : Vecteurs caractérisant les règles de branchement (a.x < b)
- b : Idem
- c : c[t] est la classe du noeud d'indice t (-1 si aucune classe n'est attribuée)

Sortie :
- this::Tree : l'arbre correspondant
""" 
function Tree(D::Int64, a::Matrix{Float64}, b::Vector{Float64}, c::Vector{Int64})
    this = Tree()
    this.D = D
    this.a = a
    this.b = b
    this.c = c

    return this
end

"""
Constructeur d'arbre univarié où la valeur de b est ajustée pour recentrer la séparation par rapport aux données

Entrées :
- D : Nombre maximal de séparations d'une branche (profondeur de l'arbre - 1)
- a : Vecteurs caractérisant les règles de branchement (a.x < b)
- c : c[t] est la classe du noeud d'indice t (-1 si aucune classe n'est attribuée)
- u : flot des données dans l'arbre
- x : caractéristiques des données d'entraînement   

Sortie :
- this::Tree : un arbre
"""
function Tree(D::Int64, a::Matrix{Float64}, c::Vector{Int64}, u::Matrix{Int64}, x::Matrix{Float64})
           
    this = Tree()
    this.D = D
    this.a = a
    this.c = c

    sepCount = 2^D - 1
    leavesCount = 2^D
    dataCount = length(x[:, 1])
    featuresCount = length(x[1, :])

    # Borne supérieure de b pour chaque séparation
    ub_b = ones(Float64, sepCount)
    
    # Borne inférieure de b pour chaque séparation
    lb_b = zeros(Float64, sepCount)
    ## Fixe lb_b et ub_b

    # Pour chaque donnée
    for i in 1:dataCount

        # Trouve la feuille où elle arrive
        # (en partant des sommets les plus bas dans l'arbre, trouver le premier sommet atteint par la donnée)
        t = sepCount+leavesCount
        while u[i, t] == 0
            t -= 1
            if t == 0
                break
            end
        end

        # Si cette donnée a un flot dans l'arbre (i.e., si elle est correctement classifiée)
        if t != 0

            # Pour chaque sommet parent a_t et tant que la racine n'est pas atteinte
            while t>1
                a_t = t ÷ 2
                if t % 2 == 0
                    lb_b[a_t] = max(lb_b[a_t], sum(a[j, a_t]*x[i, j] for j in 1:featuresCount))
                else
                    ub_b[a_t] = min(ub_b[a_t], sum(a[j, a_t]*x[i, j] for j in 1:featuresCount))
                end
                t = a_t
            end
        end
    end

    this.b = zeros(Float64, sepCount)
    for t in 1:sepCount
        if c[t] == -1
            this.b[t] = (ub_b[t] + lb_b[t])/2
        end
    end

    return this
end

"""
FONCTION SIMILAIRE A LA PRECEDENTE UTILISEE UNIQUEMENT SI VOUS FAITES DES REGROUPEMENTS 

Constructeur d'arbre univarié où la valeur de b est ajustée pour recentrer la séparation par rapport aux données

Entrées :
- D : Nombre maximal de séparations d'une branche (profondeur de l'arbre - 1)
- a : Vecteurs caractérisant les règles de branchement (a.x < b)
- c : c[t] est la classe du noeud d'indice t (-1 si aucune classe n'est attribuée)
- u : flot des données dans l'arbre
- clusters : partition des données d'entraînement   

Sortie :
- this::Tree : un arbre
"""
function Tree(D::Int64, a::Matrix{Float64}, c::Vector{Int64}, u::Matrix{Int64}, clusters::Vector{Cluster})

    this = Tree()
    this.D = D
    this.a = a
    this.c = c

    sepCount = 2^D - 1
    leavesCount = 2^D
    dataCount = length(clusters)
    featuresCount = length(clusters[1].lBounds)
    
    # Upper bound of b for each separation
    ub_b = ones(Float64, sepCount)
    
    # Lower bound of b for each separation
    lb_b = zeros(Float64, sepCount)
    ## Fixe lb_b et ub_b
    
    # Pour chaque cluster
    for i in 1:dataCount

        # Trouver la feuille où il arrive
        # (en partant des sommets les plus bas dans l'arbre, trouver le premier sommet atteint par le cluster)
        t = sepCount+leavesCount
        while u[i, t] == 0
            t -= 1
            if t == 0
                break
            end
        end

        # Si cette donnée a un flot dans l'arbre (i.e., si elle est correctement classifiée)
        if t != 0
            
            # Pour chaque sommet parent a_t et tant que la racine n'est pas atteinte
            while t>1
                a_t = t ÷ 2

                ## Warning: improve this later. Here the cluster is considered as a box. It would be better to get the distance to the separation for each data point in the cluster
                
                # If the cluster goes to the left of at
                if t % 2 == 0
                    lb_b[a_t] = max(lb_b[a_t], sum(a[j, a_t]*clusters[i].uBounds[j] for j in 1:featuresCount))

                # If the cluster goes to the right of at
                else
                    ub_b[a_t] = min(ub_b[a_t], sum(a[j, a_t]*clusters[i].lBounds[j] for j in 1:featuresCount))
                end
                t = a_t
            end
        end
    end

    this.b = zeros(Float64, sepCount)
    for t in 1:sepCount
        if c[t] == -1
            this.b[t] = (ub_b[t] + lb_b[t])/2
        end
    end

    return this
end

"""
Constructeur d'arbre multivarié où les valeurs de a et b sont ajustées pour recentrer la séparation par rapport aux données

Entrées :
- D : Nombre maximal de séparations d'une branche (profondeur de l'arbre - 1)
- c : c[t] est la classe de la donnée d'indice t (-1 si aucune classe n'est attribuée)
- u : flot des données dans l'arbre
- s_model : s[j, t] == 1 ssi le coefficient de la caractéristique j est non nul dans la séparation du sommet t
- x : caractéristiques des données d'entraînement   

Sortie :
- this::Tree : un arbre
"""
function Tree(D::Int64, c::Vector{Int64}, u::Matrix{Int64}, s_model::Matrix{Int64}, x::Matrix{Float64})
    this = Tree()
    this.D = D
    this.c = c
    
    sepCount = 2^D - 1
    dataCount = length(x[:, 1])
    featuresCount = length(x[1, :])
    
    this.a = zeros(Float64, featuresCount, sepCount)
    this.b = zeros(Float64, sepCount)

    # Pour chaque sommet interne de l'arbre
    for t in 1:sepCount

        # Si une séparation y est effectuée
        if c[t] == -1

            # Indice des données allant à droite ou à gauche
            I_R = Int64[]
            I_L = Int64[]
            for i in 1:dataCount
                if u[i, t*2] == 1
                    push!(I_L, i)
                elseif u[i, t*2+1] == 1
                    push!(I_R, i)
                end
            end

            len_l = length(I_L)
            len_r = length(I_R)

            m = Model(CPLEX.Optimizer)
            set_silent(m)
            @variable(m, a[1:featuresCount], base_name = "a_j")
            @variable(m, s[1:featuresCount], Bin, base_name = "s_j")
            @variable(m, b, base_name = "b")
            @variable(m, e[1:(len_l + len_r)], base_name = "ecart") # Ecart entre chaque variable et la séparation
            @variable(m, e_min, base_name = "ecart_min")

            @constraint(m, -1 <= b)
            @constraint(m, b <= 1)
            @constraint(m, [j in 1:featuresCount], -s[j] <= a[j])
            @constraint(m, [j in 1:featuresCount], a[j] <= s[j])
            @constraint(m, sum(s[j]  for j in 1:featuresCount) <= sum(s_model[j, t] for j in 1:featuresCount)) # on ne veut pas augmenter la "complexité" de la séparation
            @constraint(m, [i in 1:(len_l + len_r)], e[i] >= e_min)
            @constraint(m, [i in 1:len_l], e[i] == b - sum(a[j]*x[I_L[i], j] for j in 1:featuresCount))
            @constraint(m, [i in 1:len_r], e[i+len_l] == - b + sum(a[j]*x[I_R[i], j] for j in 1:featuresCount))

            @objective(m, Max, e_min)
            optimize!(m)
            
            this.b[t] = value.(b)
            for j in 1:featuresCount
                this.a[j, t] = value.(a[j])
            end
        end
    end

    return this
end

"""
FONCTION SIMILAIRE A LA PRECEDENTE UTILISEE UNIQUEMENT SI VOUS FAITES DES REGROUPEMENTS 

Constructeur d'arbres multivariés où les valeurs de a et b sont ajustées pour recentrer la séparation par rapport aux données

Entrées :
- D : Nombre maximal de séparations d'une branche (profondeur de l'arbre - 1)
- c : c[t] est la classe du cluster d'indice t (-1 si aucune classe n'est attribuée)
- u : flot des données dans l'arbre
- s_model : s[j, t] == 1 ssi le coefficient de la caractéristique j est non nul dans la séparation du sommet t
- clusters : liste des clusters regroupant les données d'entraînement

Sortie :
- this::Tree : un arbre
"""
function Tree(D::Int64, c::Vector{Int64}, u::Matrix{Int64}, s_model::Matrix{Int64}, clusters::Vector{Cluster})
    
    this = Tree()
    this.D = D
    this.c = c

    sepCount = 2^D - 1
    dataCount = length(clusters)
    featuresCount = length(clusters[1].lBounds)

    this.a = zeros(Float64, featuresCount, sepCount)
    this.b = zeros(Float64, sepCount)
    
    # Pour chaque sommet interne de l'arbre
    for t in 1:sepCount

        # Si une sépration y est effectuée
        if c[t] == -1

            # Déterminer les données qui vont à gauche ou droite
            rightData = Matrix{Float64}(undef, 0, featuresCount)
            leftData = Matrix{Float64}(undef, 0, featuresCount)
            
            for i in 1:dataCount
                if u[i, t*2] == 1
                    leftData = vcat(leftData, clusters[i].x)
                elseif u[i, t*2+1] == 1
                    rightData = vcat(rightData, clusters[i].x)
                end
            end

            len_l = size(leftData, 1)
            len_r = size(rightData, 1)

            m = Model(CPLEX.Optimizer)
            set_silent(m)
            @variable(m, a[1:featuresCount], base_name = "a_j")
            @variable(m, s[1:featuresCount], Bin, base_name = "s_j")
            @variable(m, b, base_name = "b")
            @variable(m, e[1:len_l+len_r], base_name = "ecart") # Ecart entre chaque variable et la séparation
            @variable(m, e_min, base_name = "ecart_min")

            @constraint(m, -1 <= b)
            @constraint(m, b <= 1)
            @constraint(m, [j in 1:featuresCount], -s[j] <= a[j])
            @constraint(m, [j in 1:featuresCount], a[j] <= s[j])
            @constraint(m, sum(s[j]  for j in 1:featuresCount) <= sum(s_model[j, t] for j in 1:featuresCount)) # on ne veut pas augmenter la "complexité" de la séparation
            @constraint(m, [i in 1:len_l + len_r], e[i] >= e_min)
            @constraint(m, [i in 1:len_l], e[i] == b - sum(a[j]*leftData[i, j] for j in 1:featuresCount))
            @constraint(m, [i in 1:len_r], e[i+len_l] == - b + sum(a[j]*rightData[i, j] for j in 1:featuresCount))

            @objective(m, Max, e_min)
            optimize!(m)

            this.b[t] = value.(b)
            for j in 1:featuresCount
                 this.a[j, t] = value.(a[j])
            end
        end
    end

    return this
end
