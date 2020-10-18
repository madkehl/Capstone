from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance


def prep_for_rf(response_col, df, colsub = 5, nested = 5):
    '''
    INPUT:
    y var/response col, the dataframe, a substring (of related variables one might want to omit, 
    in this case the sub-components of total review scores), and possibly a nested var.
    
    Defaults are set to arbitrary numeric values for colsub and nested for the isinstance if statements 
    
    OUTPUT:
    X and ys that have had nans properly addressed
    
    STEPS:
        1. This function drops all rows that are missing the dependent variable/y variable.
        2. If colsub has a value, it drops columns with that regex
        3. If nested has a value, it groups by this value and finds the mean within the nested value
        (along the lines of person-mean imputation).  This is to try to improve the accuracy of the
        mean imputation, and preserve differences between people. So in this case, if a listing appears
        twice, and the value for beds is for some reason missing in one, it is imputed with the average 
        # of beds (Nas ignored by default) for this listing, not the average number of beds overall.
        4. Finally, if there is no other option, it will fill nas with the general mean
        5. IF there is no nested variable, it will simply fill the nas with the mean.
        6. returns as X, y
    '''
    fillmean = lambda col: col.fillna(col.mean())
    df2 = df.copy()
    df2 = df2.dropna(subset = [response_col], axis = 0).reset_index(drop = True)
    if isinstance(colsub, str):
        X = df2[df2.columns.drop(list(df2.filter(regex=colsub)))]
        X = X.reset_index(drop = True)
    if isinstance(nested, str):
        X = X.groupby([nested]).transform(lambda x: x.fillna(x.mean()))
        X[nested] = df2[nested]
        X = X.apply(fillmean)
    else:
        X = X.apply(fillmean)
    y = df2[response_col]
    return(X, y)

def run_models_kfold(function, X, y, kfoldn = 10, n_rep = 15, pi_bool = True):
    '''
    INPUT:
    scikit rf regressor model, x vals, y vals, number of kfold splits, # of permutations,
    boolean whether or not to find permutation importances for kfold across trials
    
    OUTPUT:
    mae scores, mse scores, feature ranking
    
    STEPS:
        1. Initialize the accuracy of the models to blank lists. Respective vals will be appended to this list
        2. iterate over kfold splits
        3. if pi_bool == True 
            a. then use scikit's permutation importance to see which features are most important
            b. concatenate into labeled df.
            c. transform into long format
            d. label the sum of scores as 'importance' and indexers as 'labels'(will feed into further functions better in this format)
        4. return list containing (1. list of mean absolute error, 2. list of mean squared error, 3. list of dfs of permutation importance)
            
      
    '''
    kf = KFold(n_splits=kfoldn,shuffle=True, random_state = 15)
    mae_ = []
    mse_ = []
    per_imps = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = function.fit(X_train, y_train)
        mae_.append(mean_absolute_error(y_test, model.predict(X_test)))
        mse_.append(mean_squared_error(y_test, model.predict(X_test)))
   
        if pi_bool == True:
            result = permutation_importance(function, X_test, y_test, n_repeats=n_rep,
                                random_state=15, n_jobs=2)
            sorted_idx = result.importances_mean.argsort()
            perm_imp = pd.DataFrame(result.importances[sorted_idx].T)
            perm_imp.columns = X_test.columns[sorted_idx]
            perm_imp = perm_imp.T.rename_axis("labels")
            perm_imp['importance'] = perm_imp.mean(axis = 1)
            perm_imp = perm_imp.reset_index()
            
            per_imps.append(perm_imp)
    return([mae_, mse_, per_imps])

def run_each_participant(function, X, y, feat_bool = True):
    '''
    INPUT:
    scikit rf regressor model, x vals, y vals, boolean whether or not to find feature importances
    
    OUTPUT:
    mae scores, mse scores, feature ranking
    '''
    X = X.copy()
    y = y.copy()
    accuracy_model = []
    mse_ = []
    feat_imps = []
    
    for i in range(0,len(X[X.columns[0]])):
        
        X_test = X.loc[i]
        y_test = pd.Series(y[i])
        X_test = X_test.values.reshape(1, -1)
        
        X_train = X.drop(X.index[i])
        y_train =np.delete(y, i)
        model = function.fit(X_train, y_train)
        
        accuracy_model.append(mean_absolute_error(y_test, model.predict(X_test)))
        mse_.append(mean_squared_error(y_test, model.predict(X_test)))
        if feat_bool == True:
            feat_imp = pd.DataFrame({
    
                'labels': X_train.columns,
                'importance':model.feature_importances_
            })

            feat_imp = feat_imp.reset_index(drop = True)
            feat_imp = feat_imp.sort_values(by = 'importance', axis = 0)
            feat_imps.append(feat_imp)
    
    return([accuracy_model, mse_, feat_imps])

def specify_best(model, X, y, typ = 'RF'):
    '''
    INPUT:
    scikit regressor model, x vals, y vals, str telling if it is rf or svr
    
    OUTPUT:
    best parameter values for min samples split, max features, or kernel gamma and C
    '''
    if typ == 'RF':
        params =  {'min_samples_split': np.arange(2, 100, 20), 'max_features': np.arange(2,10,2)}
    elif typ == 'svr':
         params =  {'kernel': ['rbf', 'linear', 'sigmoid'], 'gamma': np.arange(0.001,0.9,0.1), 'C': [0.0001,0.01,1,10,100]}
    clf = GridSearchCV(model, params,cv = 5)
    clf.fit(X, y)
    print(clf.best_params_)
    return(clf.best_params_)

def top_features(feat_imps, featn = 15, mean = False):
    '''
    INPUT:
    result df from run_each_participant, number of features you want returned, or if you want cutoff to be mean
    only can use one at a time.
    
    OUTPUT:
    most important features, their importances, and frequency of models that they appeared in (as greater than mean or top featn)
    
    STEPS:
        1. iterate through feat_imps, creat index var showing which trial it came from and reformat
        2. concatenate this into one dataframe
        3. fill nas (instances where a feature didn't occur) with zero (same as feature not occuring)
        4. create a series of mean_vals
        5. whichever selection method you chose, will use this series to select the topn or the > mean features
        6. using the names/index from this series it will count how often they occur
        6. create and return new dataframe.
    '''
    all_feat = []
    count = 0
    
    for i in feat_imps:
        i = i.reset_index(drop = False)
        i['index'] = [count]*len(i['index'])
        i = i.pivot(index = 'index', columns='labels', values='importance')
        all_feat.append(i)
        count += 1
    
    all_feat_df = pd.concat(all_feat, axis = 0)
    all_feat_df_filled = all_feat_df.fillna(0)
    mean_feat = all_feat_df_filled.mean(axis = 0)
    if mean == False:
        top_feat = mean_feat.sort_values().iloc[len(mean_feat)-featn: len(mean_feat)]
    else:
        top_feat = mean_feat.sort_values()
        top_feat = top_feat.where(top_feat > np.mean(top_feat.values))
        top_feat = top_feat.dropna()
        
    names = top_feat.index
    count_feat = all_feat_df.count(axis = 0)
    count_feat_sub = count_feat.loc[list(names)]
    
    top_feat_df = pd.DataFrame({
        'features':names,
        'feat_imps':top_feat.values,
        'count_models': count_feat_sub.values
    })
    return(top_feat_df)
    
def print_vals(best_param, run_list, top_feat_df):
    '''
    INPUT:
    set of results from above functions
    
    OUTPUT:
    prints them into notebook to give an initial idea of what results are (not intended to be final form)
    '''
    print('mss: ' + str(best_param.get('min_samples_split')))
    print('max_feat: ' + str(best_param.get('max_features')))
    print('mean_mae: ' + str(np.mean(run_list[0])))
    print('mean_mse: '+ str(np.mean(run_list[1])))
    print('\n' + 'frequency: ' + '\n')
    for i in list(top_feat_df['count_models']):
        print(i)  
    print('\n' + 'feat_imp: ' + '\n')
    for i in list(top_feat_df['feat_imps']):
        print(i)
    print('\n' + 'top_ features' + '\n')
    for i in list(top_feat_df['features']):
        print(i)