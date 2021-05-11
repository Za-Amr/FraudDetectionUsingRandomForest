import numpy as np
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve 
from sklearn.metrics import recall_score,precision_recall_curve, average_precision_score,precision_score,f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from time import time

def get_model_results(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, model,model_name):
    """
    model: sklearn model (e.g. RandomForestClassifier)
    """
    # Fit training model 
    model.fit(X_train, y_train)

    # Obtain the predicted values and probabilities from the model 
    predicted = model.predict(X_test)
    probs = model.predict_proba(X_test)
    print ("\033[107m" + "Model : "+ str(model_name) +   "\033[0m")
    print("-------------------------------------------------------")
    
    #try:
     #   probs = model.predict_proba(X_test)
      #  print('The ROC Score = %0.3f '% roc_auc_score(y_test, probs[:,1]))
    #except AttributeError:
     #   pass

    Recall=recall_score(y_test, predicted)
    Precision=precision_score(y_test, predicted)
    F1=f1_score(y_test, predicted)
    ROC_Score=roc_auc_score(y_test, probs[:,1])
    Confusion_test=confusion_matrix(y_test, model.predict(X_test))
    Confusion_Normalized_test=confusion_matrix(y_test, model.predict(X_test),normalize='true' )
    Confusion_train=confusion_matrix(y_train, model.predict(X_train))
    Confusion_Normalized_train=confusion_matrix(y_train, model.predict(X_train),normalize='true' )
    
    print('Accuracy: {:.3f}'.format(accuracy_score(y_test, predicted)))
    print('Precision: {:.3f}'.format(Precision))
    print('Recall: {:.3f}'.format(Recall))
    print('F1: {:.3f}'.format(F1))

    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

    print('The ROC Score = %0.3f '% ROC_Score)
    print('\nClassification Report - Test set:')
    print('-----------------------------------------')
    print(classification_report(y_test, predicted,target_names=['Non Fraud', 'Fraud']))

    print('\nConfusion Matrix in number - Test set:')
    print('-----------------------------------------')
    print( Confusion_test)

    print('\nNormalized Confusion Matrix -Test set :')
    print('-----------------------------------------')
    print( Confusion_Normalized_test)
    print('\nClassification Report - Training set:')
    print('-----------------------------------------')
    print(classification_report(y_train, model.predict(X_train),target_names=['Non Fraud', 'Fraud']))
    print('\nConfusion Matrix - Training set:')
    print('-----------------------------------------')
    print(Confusion_train)
    print('\nNormalized Confusion Matrix - Training set:')
    print('-----------------------------------------')
    print(Confusion_Normalized_train)
    
    dict_model= {"Model":model_name, "Recall":Recall,"Precision":Precision, "F1":F1,
                 "FN_rate":Confusion_Normalized_test[1,0]*100,
                 "FP_rate":Confusion_Normalized_test[0,1]*100,
                 "AUCROC":ROC_Score}
    
    return dict_model
    
    
def grid_model(clf, parameters, X_train, y_train, X_test,y_test, model_name):
 

    # grid search for parameters of the model and recall for the optimisation
    grid_search = GridSearchCV(clf, parameters, n_jobs=-1,scoring='recall', verbose=1, cv=5)
    
    print ("\033[107m" + "Model : "+ str(model_name) +   "\033[0m")
    print("-------------------------------------------------------")
    print("Performing grid search...")
    print("parameters:")
    
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("Time : done in %0.2fs" % (time() - t0))
    print()

    print("Best Cross Validation Recall: %0.2f" % grid_search.best_score_)
    print("\n")
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    probs = grid_search.best_estimator_.predict_proba(X_test)
    
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    print("\n")
    print('ROC Score for the test set: ' , roc_auc_score(y_test, probs[:,1]))
    print("-----------------------------------------------------------------")
    print("\n")
    print(" Classification Report - Test Data ")
    print("----------------------------------------")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test),target_names=['Non Fraud', 'Fraud']))
    print("\n")
    print(" The confusion matrix in number - Test Data ")
    print("----------------------------------------------------")
    print(confusion_matrix(y_true=y_test, y_pred=grid_search.best_estimator_.predict(X_test)))
    print("-----------------------------------------------------")
    print(" The normalized confusion matrix - Test Data")
    print(confusion_matrix(y_true=y_test, y_pred=grid_search.best_estimator_.predict(X_test),normalize='true'))
    print("\n")
    print(" Classification Report - Train Data  ")
    print("----------------------------------------")
    print(classification_report(y_train, grid_search.best_estimator_.predict(X_train),target_names=['Non Fraud', 'Fraud']))       
    print("\n")
    print(" The confusion matrix in number - Train Data")
    print("-------------------------------------------")
    print(confusion_matrix(y_true=y_train, y_pred=grid_search.best_estimator_.predict(X_train)))
    print(" The normalized confusion matrix - Train Data")
    print("---------------------------------------------------")
    print(confusion_matrix(y_true=y_train, y_pred=grid_search.best_estimator_.predict(X_train),normalize='true'))
    
    return     
    
    
    
    