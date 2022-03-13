import pandas as pd
import joblib
from sklearn import model_selection
import sklearn.metrics as metrics
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold 
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


base_data = pd.read_csv('/Users/marcelo/Desktop/Rita/Tese/tese.nosync/Data/Infeatures/InFeatures_data-filtered-random-final.csv', sep=';')

# Mix dataset
shuffle_data = base_data.sample(frac=1)

# Split In and Out features
columns_len = len(shuffle_data.columns)

in_features_data = shuffle_data.iloc[:, :(columns_len-1)]
out_feature_data = shuffle_data.iloc[:, (columns_len-1):]

# Pre-Processing dataset

# Transform and uniformizate data - https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
fit_input = SimpleImputer().fit_transform(in_features_data)

# Removes all low-variance features - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
filtered_input = VarianceThreshold().fit_transform(fit_input)    

filtered_output = out_feature_data.values.ravel()

# Load models data
NB_model= BaggingClassifier(GaussianNB())
tree_model = BaggingClassifier(ExtraTreesClassifier())
knn = BaggingClassifier(KNeighborsClassifier())
logistic = BaggingClassifier(linear_model.LogisticRegression())
GB_model= GradientBoostingClassifier()
RF_model= BaggingClassifier(RandomForestClassifier())
svm_model = svm.SVC()

# pipeknn=make_pipeline(preprocessing.StandardScaler(),knn)
# pipelogistic=make_pipeline(preprocessing.StandardScaler(),logistic)
# pipesvm=make_pipeline(preprocessing.StandardScaler(),svm_model)

## Generate Evaluation

# models_with_labels = zip([NB_model,tree_model,knn,logistic,GB_model,RF_model,svm_model],["Naive Bayes","ExtraTreeClassifier","K nearest neighbours","Logistic Regression","GradientBoosting","RandomForest","SVM"])
models_with_labels = zip([NB_model,tree_model,knn,logistic,GB_model,RF_model,svm_model],["GradientBoosting","RandomForest","SVM"])


file = open("BestModels.txt", "w")

for model,label in models_with_labels:
    # Accuracy
    scores = model_selection.cross_val_score(model, filtered_input, filtered_output, cv=5, scoring="accuracy")
    file.write("Accuracy: %0.2f (+/- %0.2f) [%s]\n" % (scores.mean(),scores.std(),label))
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(),scores.std(),label))

    # Precision
    scores = model_selection.cross_val_score(model, filtered_input, filtered_output, cv=5, scoring="precision")
    file.write("Precision: %0.2f (+/- %0.2f) [%s]\n" % (scores.mean(),scores.std(),label))
    print("Precision: %0.2f (+/- %0.2f) [%s]" % (scores.mean(),scores.std(),label))

    # Recall
    scores = model_selection.cross_val_score(model, filtered_input, filtered_output, cv=5, scoring="recall")
    file.write("Recall: %0.2f (+/- %0.2f) [%s]\n" % (scores.mean(),scores.std(),label))
    print("Recall: %0.2f (+/- %0.2f) [%s]" % (scores.mean(),scores.std(),label))

    # F1
    scoresf1 = model_selection.cross_val_score(model, filtered_input, filtered_output, cv=5, scoring="f1")
    file.write("F1: %0.2f (+/- %0.2f) [%s]\n" % (scoresf1.mean(),scoresf1.std(),label))
    print("F1: %0.2f (+/- %0.2f) [%s]" % (scoresf1.mean(),scoresf1.std(),label))

    # ROC-AUC
    scoresAUC= model_selection.cross_val_score(model, filtered_input, filtered_output, cv=5, scoring="roc_auc")
    file.write("ROC-AUC: %0.2f (+/- %0.2f) [%s]\n" % (scoresAUC.mean(),scoresAUC.std(),label))
    print("ROC-AUC: %0.2f (+/- %0.2f) [%s]" % (scoresAUC.mean(),scoresAUC.std(),label))

    # MCC
    scoresAUC= model_selection.cross_val_score(model, filtered_input, filtered_output, cv=5, scoring=metrics.make_scorer(metrics.matthews_corrcoef))
    file.write("MCC: %0.2f (+/- %0.2f) [%s]\n" % (scoresAUC.mean(),scoresAUC.std(),label))
    print("MCC: %0.2f (+/- %0.2f) [%s]" % (scoresAUC.mean(),scoresAUC.std(),label))

file.close()

## Generate Models

# #Saving the NB model 
# NBM = NB_model.fit(filtered_input, filtered_output)
# learned_NB = "NB_model.pkl"
# joblib.dump(NBM,learned_NB)
        
# #Saving the Tree model
# treeM = tree_model.fit(filtered_input, filtered_output)
# learned_tree = "Tree_model.pkl"
# joblib.dump(treeM,learned_tree)
            
# #Saving the knn model 
# knnM = knn.fit(filtered_input, filtered_output)
# learned_knn = "Knn_model.pkl"
# joblib.dump(knnM,learned_knn)
        
# #Saving the logistic regression model
# logisticM = logistic.fit(filtered_input, filtered_output)
# learned_logistic = "Logistic_model.pkl"
# joblib.dump(logisticM,learned_logistic)
    
        
# #Saving the GB model 
# GBM = GB_model.fit(filtered_input, filtered_output)
# learned_GB = "GB_model.pkl"
# joblib.dump(GBM,learned_GB)
        
# #Saving the RF model 
# RFM = RF_model.fit(filtered_input, filtered_output)
# learned_RF = "RF_model.pkl"
# joblib.dump(RFM,learned_RF)
            
# #Saving the SVM model
# svmM = svm_model.fit(filtered_input, filtered_output)
# learned_svm = "SVM_model.pkl"
# joblib.dump(svmM,learned_svm)
