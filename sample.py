
##* objective functions for multiple algorithms
def objective_cpu(trial):
    train_data, test_data, features, target, label_names = load_data()
    import secrets	# using built-in func. over py_3.6.3
    algorithm_names = ['RF', 'SVM']
    algorithm_name = secrets.choice(algorithm_names)

    if algorithm_name == 'RF':
        RF_cv = trial.suggest_int("RF_cv", 5, 5)
        RF_n_estimators = trial.suggest_int("RF_n_estimators", 
                203,1909)
        RF_criterion = trial.suggest_categorical("RF_criterion", ['gini', 'entropy'])
        RF_min_samples_split = trial.suggest_float(
            "RF_min_samples_split", 0.257, 0.971)
        RF_max_features = trial.suggest_float("RF_max_features",0.081, 0.867)
        RF_min_samples_leaf = trial.suggest_float("RF_min_samples_leaf", 0.009, 0.453)
        # integrated algorithm list
        trial.set_user_attr('algorithm_name', algorithm_name)
        
        train_fetched = fetch_images(train_data['file_loc'], 'GRAYSCALE', 1)
        test_fetched = fetch_images(test_data['file_loc'], 'GRAYSCALE', 1)
        features = ['pixel_'+str(x) for x in range(train_fetched.shape[1])]
        train_data = pd.concat([train_data, pd.DataFrame(train_fetched, columns=["pixel_"+str(x) for x in range(train_fetched.shape[1])], index=train_data.index)], axis=1)
        test_data = pd.concat([test_data, pd.DataFrame(test_fetched, columns=["pixel_"+str(x) for x in range(test_fetched.shape[1])], index=test_data.index)], axis=1)
        
        ##* Random Forests Classification
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = RF_n_estimators, criterion = RF_criterion, min_samples_split = RF_min_samples_split, max_features = RF_max_features, min_samples_leaf = RF_min_samples_leaf)
        # K-fold cross-validation, Number of trees, Quality of a split, The min. num required to split an internal node, The number of features to consider when looking for the basic split:, The max. num required to be at a leaf node
        ## Cross validation score
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(clf, train_data[features], train_data[target], cv = RF_cv, scoring='f1_macro', n_jobs = 15)
        
        clf.fit(train_data[features], train_data[target])
        ## results
        
        ##* Predict using the model we made.
        predicted = clf.predict(test_data[features])
        confidence = metrics.f1_score(predicted, test_data[target], average='macro')	# Returns mean f1-score.

        sdroptim.retrieve_model(algorithm_name, clf, trial.number, scores.mean(), top_n_all = 10, top_n_each_algo = 5)
        return scores.mean()
