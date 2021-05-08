#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score,                                    KFold, train_test_split, cross_validate, ParameterGrid
from sklearn.base import BaseEstimator, TransformerMixin,  clone

def train_hold_test_split(features, target, tr_size, ho_size, shuffle, random_state, stratify, use_test):
    
    '''
    Разделение данных на 3 части
    
    Параметры:
        1) tr_size        | доля тренировочной части          | float 
        2) ho_size        | доля отложенной части             | float 
        3) shuffle        | перемешивание                     | bool
        4) random_state   | генератор случайных чисел         | int
        5) stratify       | стратификация целевой переменной  | bool
        6) use_test       | исользовать тестовую часть        | bool
    
    '''
    
    if use_test:
        X_trho, X_te, y_trho, y_te = train_test_split(X, y,
                                                      train_size = tr_size,
                                                      shuffle = shuffle,
                                                      random_state = random_state,\
                                                      stratify = y if stratify else None)
            
        X_trho, X_te, y_trho, y_te = train_test_split(X_trho, y_trho,
                                                      train_size = tr_size,
                                                      shuffle = shuffle,
                                                      random_state = random_state,\
                                                      stratify = y_trho if stratify else None)
        return (X_tr, X_ho, X_te, y_tr, y_ho, y_te)
    else:
        X_tr, X_ho, y_tr, y_ho = train_test_split(X, y,                                                  train_size = tr_size,                                                  shuffle = shuffle,
                                                  random_state = random_state,\
                                                  stratify = y if stratify else None)
        return (X_tr, X_ho, y_tr, y_ho)
    
def convert_dtypes(df):
    '''
    преобразование типов
    '''
    if not(isinstance(df, pd.DataFrame)):
        raise TypeError('df should be of type pd.DataFrame')        
    df_c = df.copy()
    for col in df.columns:
        ser = df[col]
        try:
            ser2 = ser.astype('datetime64')
            df_c[col] = ser2
        except:
            try:
                ser2 =ser.astype(int)
                if (ser != ser2).any():
                    try:
                        df_c[col] =ser.astype(float)
                    except:
                        pass
                    
            except:
                pass
        
    try:
        return pd.concat([df_c.select_dtypes('datetime64'), df_c.select_dtypes(exclude = ['datetime64'])], 1)
    except:
        return df_c
    
class SklearnHelperTargetEncoder(BaseEstimator, TransformerMixin):
    '''
    Кодирование категорий с помощью целевой переменной
    
    Параметры:
        1) n_iter           | число итераций                | int
        2) n_folds          | число фолдов                  | int
        3) min_samples_leaf | минимальный размер категории, | int
                            | необходимый для учета значения|
                            | целевого признака             | 
        4) seed             | генератор случайных чисел     | int
        
    
    '''
    def __init__(self, n_iter, n_folds, min_samples_leaf, seed):
        self.n_iter = n_iter
        self.n_folds = n_folds
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
    def fit(self, X, y=None):
        self.y_mean = y.mean()
        _df_tr = pd.concat([X, y], 1)
        target_col = _df_tr.columns[-1]
        to_encode = _df_tr.columns[:-1]
        
        L_tr = []        
        self.L_d_encs = []
        for i in tqdm_notebook(range(self.n_iter)): 
            enc_tr = pd.DataFrame(index = _df_tr.index, columns = to_encode).fillna(0.0)
            for col in to_encode:
                for tr_idx, val_idx in KFold(self.n_folds, shuffle = True,random_state = self.seed+i)                                       .split(_df_tr):                    
                    grp = _df_tr.iloc[tr_idx].groupby(col)[target_col].agg({'mean', 'count'}) 
                    d_enc = grp[grp['count']>=self.min_samples_leaf]['mean'].to_dict()
                    self.L_d_encs.append((col, d_enc))
                    to_enc_tr =_df_tr.iloc[val_idx]                    
                    enc_tr.loc[to_enc_tr.index, col] = to_enc_tr[col].map(d_enc)                  
            L_tr.append(enc_tr)    
            
        self.enc_tr =  pd.concat(L_tr, 1)
        self._df_tr = _df_tr
        return self    
    def transform(self, X):
        if np.all(X.values == self._df_tr.values):
            return self.enc_tr.fillna(self.y_mean) 
        else:
            df_enc = pd.DataFrame(index = X.index, columns=X.columns).fillna(0.0)
            for feat, d in tqdm_notebook(self.L_d_encs):
                df_enc.loc[:, feat] += X[feat].map(d) / self.n_iter
            return df_enc.fillna(self.y_mean)

class SklearnHelperFeatureSelector(BaseEstimator, TransformerMixin):
    ''' 
    Отбор признаков (последовательное добавление)
    
    Параметры:
        1) model         | модель                                     | 
        2) cv            | схема валидации                            | 
        3) scoring       | метрика качества                           | 
        4) show_progress | печатать текущее значение метрики качества | bool
    
    '''
    def __init__(self, model, cv, scoring, show_progress):
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.show_progress = show_progress
    def fit(self, X, y=None):
        try:
            _X = X.todense()
        except:
            _X =X.copy()            
        cv_scores = []
        for i in tqdm_notebook(range(_X.shape[1])):
            _X_curr = _X[:, i].reshape(-1,1)
            mean_cv_score = cross_val_score(self.model, _X_curr, y, cv =self.cv, scoring = self.scoring, n_jobs=-1).mean()
            
            cv_scores.append(mean_cv_score)
        order = np.argsort(cv_scores)[::-1]
        to_drop_before, best_features, best_cv_score = [], [], -np.inf
        for i in tqdm_notebook(order):
            curr_features = best_features+[i]
            _X_curr = _X[:, curr_features]
            mean_cv_score = cross_val_score(self.model, _X_curr, y, cv =self.cv, scoring = self.scoring, n_jobs=-1).mean()
            if mean_cv_score>best_cv_score:
                best_cv_score = mean_cv_score
                best_features = curr_features
                if self.show_progress:
                    print('new best score = {:.5f}'.format(best_cv_score))
            else:
                to_drop_before.append(i)
        while True:
            to_drop_after = []
            for i in tqdm_notebook(to_drop_before):
                curr_features = best_features+[i]
                _X_curr = _X[:, curr_features]
                mean_cv_score = cross_val_score(self.model, _X_curr, y, cv =self.cv, scoring = self.scoring, n_jobs=-1).mean()
                if mean_cv_score>best_cv_score:
                    best_cv_score = mean_cv_score
                    best_features = curr_features
                    if self.show_progress:
                        print('new best score = {:.5f}'.format(best_cv_score))
                else:
                    to_drop_after.append(i)
            if to_drop_before == to_drop_after:
                break
            else:
                to_drop_before = to_drop_after  
        self.best_features = best_features
        self.best_cv_score = best_cv_score
    def transform(self, X):
        if isinstance(X, csc_matrix):
            _X = X.copy()
        else:            
            _X = csc_matrix(X) 
        return _X[:, self.best_features]
    
    

class SklearnHelperMetaFeaturesRegressor(TransformerMixin, BaseEstimator):   
    ''' 
    Получение метапризнаков с помощью базовых моделей
    
    Параметры:
        1) base_model     | модель                             | 
        2) nfolds         | число фолдов                       | int
        3) seed           | генератор случайных чисел          | int
        4) path_to_folder | путь к папке с обученными моделями | os path
    
    '''
    def __init__(self, base_model, nfolds, seed, path_to_folder):
        self.base_model = base_model   
        self.nfolds = nfolds
        self.kf = KFold(nfolds, random_state = seed, shuffle = True)
        self.path_to_folder=path_to_folder
    def fit(self, X, y=None):        
        if not os.path.exists(self.path_to_folder):
            os.makedirs(self.path_to_folder)
        else:
            shutil.rmtree(self.path_to_folder)
            os.makedirs(self.path_to_folder)
            
        self.X = X
        self.Z_tr = np.zeros((len(y), 1))
        for i, (tr_idx, val_idx) in tqdm_notebook(enumerate(self.kf.split(X, y)),                                                  total = self.kf.n_splits):
            # обучаем модель на тренировочной части
            self.base_model.fit(X[tr_idx], y[tr_idx])
            path_to_model= os.path.join(self.path_to_folder, f'model_{i}.pkl')            
            joblib.dump(self.base_model, path_to_model)               
            # предсказываем валидационную
            self.Z_tr[val_idx, 0] = self.base_model.predict(X[val_idx]) 
        return self
    def predict(self, X):
        if isinstance(self.X, (np.ndarray, np.generic)):
            if np.array_equal(self.X, X, equal_nan=True):
                return self.Z_tr.flatten()
        elif isinstance(self.X, (csc_matrix)):
            if np.array_equal(self.X[:,0].toarray().flatten(), X[:,0].toarray().flatten(), equal_nan=True):
                return self.Z_tr.flatten()
        predictitons=np.zeros((X.shape[0], self.nfolds))
        for i, filename in enumerate(os.listdir(self.path_to_folder)):
            path_to_model= os.path.join(self.path_to_folder, filename)
            fitted_model = joblib.load(path_to_model)   
            predictitons[:, i] = fitted_model.predict(X)
        return np.mean(predictitons, 1).flatten()
            
        
class SklearnHelperRegressorValidator(BaseEstimator, TransformerMixin):
    ''' 
    Валидация модели
    
    Параметры:
        1) model           | модель                       | 
        2) cv              | схема валидации              | 
        3) cv_scoring      | оценщик метрики (валидация)  | 
        4) ho_scoring_func | оценщик метрики (отложенная) | 
        5) to_tune         | оптимизация гиперпараметров  | bool
    
    '''
    def __init__(self, model, cv, cv_scoring, ho_scoring_func, to_tune=True):
        self.model = model
        self.cv = cv
        self.cv_scoring = cv_scoring
        self.ho_scoring_func = ho_scoring_func
        self.to_tune = to_tune
    def fit(self, X_tr, y_tr, X_ho, y_ho):
        #######################################################################################################
        def _hp_tune_v1(model, grid, X, y, cv, scoring):    
            gs = GridSearchCV(model,param_grid=grid,cv = cv, scoring = scoring, n_jobs=-1, verbose = 1)
            gs.fit(X, y)
            best_estimator_ = clone(gs.best_estimator_)
            del gs
            gc.collect()    
            return best_estimator_

        def _hp_tune_v2(model, grid1, grid2, grid3, X_tr, y_tr, X_ho, y_ho, cv, scoring): 
            fit_params={'early_stopping_rounds':10,                        'eval_set':[(X_ho, y_ho)],                        'verbose':0}
            gs = GridSearchCV(model, param_grid=grid1, cv = cv, scoring=scoring, n_jobs=-1, verbose=1)
            gs.fit(X_tr, y_tr, **fit_params)    
            bp = gs.best_params_
            model = model.set_params(**bp)
            del gs
            gc.collect()

            gs = GridSearchCV(model,param_grid = grid2, cv = cv, scoring = scoring, n_jobs=-1, verbose = 1)
            gs.fit(X_tr, y_tr, **fit_params)    
            bp.update(gs.best_params_)
            model = model.set_params(**bp)
            del gs
            gc.collect()
            bp_c = bp.copy()

            best_score = -np.inf
            for params in tqdm_notebook(list(ParameterGrid(grid3))):
                bp_c.update(params)
                model = model.set_params(**bp_c)
                mean_cv_score = cross_val_score(model, X_tr, y_tr, cv=cv, scoring =scoring, n_jobs=-1).mean()
                if mean_cv_score>best_score:
                    best_score = mean_cv_score            
                    best_estimator_ = model
                else:
                    break
            return clone(best_estimator_)        
        ##############################################################################################################
        
        if not(self.to_tune):
            self.best_model = self.model
            self.mean_cv_score = cross_val_score(self.best_model, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring).mean()
            self.best_model.fit(X_tr, y_tr)
            if self.predict_proba:
                self.ho_score = self.ho_scoring_func(y_ho, self.best_model.predict_proba(X_ho)[:, 1])
            else:
                self.ho_score = self.ho_scoring_func(y_ho, self.best_model.predict(X_ho))                                             
        else:
            if type(self.model).__name__ in ('DecisionTreeRegressor', 'ExtraTreeRegressor'):
                tree_pg = {'max_depth':np.arange(7, 41), 'min_samples_leaf':[2, 20, 200]}
                self.best_model = _hp_tune_v1(self.model, tree_pg, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring)
                
                self.mean_cv_score = cross_val_score(self.best_model, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring).mean()
                self.best_model.fit(X_tr, y_tr)
                self.ho_score = self.ho_scoring_func(y_ho, self.best_model.predict(X_ho))
            elif type(self.model).__name__ in ('RandomForestRegressor', 'RandomForestClassifier',                                               'ExtraTreesRegressor', 'ExtraTreesClassifier'):
                init_params = self.model.get_params()
                trees_pg = {'max_depth':np.arange(5, 21),'min_samples_leaf':[2, 20],'n_estimators':[10],                            'n_jobs':[-1], 'random_state':[init_params['random_state']]}
                self.best_model = _hp_tune_v1(self.model,trees_pg, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring)
                bp = init_params
                best_par = self.best_model.get_params()
                del best_par['n_estimators']
                bp.update(**best_par)
                self.best_model = self.best_model.set_params(**bp)
                
                self.mean_cv_score = cross_val_score(self.best_model, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring).mean()                
                self.best_model.fit(X_tr, y_tr)
                self.ho_score = self.ho_scoring_func(y_ho, self.best_model.predict(X_ho))
                
            elif type(self.model).__name__ in ('LGBMRegressor', 'LGBMClassifier'): 
                init_params = self.model.get_params()
                lgb_grid1 = {'n_estimators':[10], 'n_jobs':[-1], 'random_state':[init_params['random_state']],                             'max_depth':np.arange(4, 21).tolist(),                             'num_leaves':[32, 64, 128, 256, 512, 1024],                             'min_child_samples':[20, 50]}
                lgb_grid2 = {'subsample':np.linspace(.1, 1, 10),                             'colsample_bytree':np.linspace(.1, 1, 10)}
                lgb_grid3 = {'learning_rate':np.linspace(.01, .1, 10), 'n_estimators':[init_params['n_estimators']]}
    
                self.best_model = _hp_tune_v2(LGBMRegressor(n_jobs=-1),                                            lgb_grid1, lgb_grid2, lgb_grid3,                                            X_tr, y_tr, X_ho, y_ho,                                            cv=self.cv, scoring=self.cv_scoring)
                lr = self.best_model.get_params()['learning_rate']
                gs = GridSearchCV(self.best_model,                               param_grid = {'learning_rate':np.linspace(lr-.09,lr+.09, 10)},                               cv=self.cv, scoring=self.cv_scoring, verbose = 1)
                gs.fit(X_tr, y_tr)
                self.best_model = gs.best_estimator_                    
                self.mean_cv_score = cross_val_score(self.best_model, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring).mean()                
                self.best_model.fit(X_tr, y_tr)
                self.ho_score = self.ho_scoring_func(y_ho, self.best_model.predict(X_ho))
                
            elif type(self.model).__name__ in ('XGBRegressor', 'XGBClassifier'): 
                init_params = self.model.get_params()
                xgb_grid1 = {'n_estimators':[10], 'n_jobs':[-1], 'random_state':[init_params['random_state']],                             'max_depth':np.arange(4, 21).tolist(),                             'min_child_weight':[20, 50]}
                xgb_grid2 = {'subsample':np.linspace(.1, 1, 10),                             'colsample_bytree':np.linspace(.1, 1, 10)}
                xgb_grid3 = {'learning_rate':np.linspace(.01, .1, 10), 'n_estimators':[init_params['n_estimators']]}
    
                self.best_model = _hp_tune_v2(XGBRegressor(n_jobs=-1),                                            xgb_grid1, xgb_grid2, xgb_grid3,                                            X_tr, y_tr, X_ho, y_ho,                                            cv=self.cv, scoring=self.cv_scoring)
                lr = self.best_model.get_params()['learning_rate']
                gs = GridSearchCV(self.best_model,                               param_grid = {'learning_rate':np.linspace(lr-.09,lr+.09, 10)},                               cv=self.cv, scoring=self.cv_scoring, verbose = 1)
                gs.fit(X_tr, y_tr)
                self.best_model = gs.best_estimator_                    
                self.mean_cv_score = cross_val_score(self.best_model, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring).mean()                
                self.best_model.fit(X_tr, y_tr)
                self.ho_score = self.ho_scoring_func(y_ho, self.best_model.predict(X_ho))
            elif type(self.model).__name__ in ('LinearRegression', 'Lasso', 'Ridge'):
                pg = {'alpha':[.1, .3, .7, 1, 10, 30, 70]}
                self.best_model = _hp_tune_v1(self.model, pg, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring)
                
                self.mean_cv_score = cross_val_score(self.best_model, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring).mean()
                self.best_model.fit(X_tr, y_tr)
                self.ho_score = self.ho_scoring_func(y_ho, self.best_model.predict(X_ho))
            elif type(self.model).__name__ in ('LinearSVR'):
                pg = {'C':[.1, .3, .7, 1, 10, 30, 70]}
                self.best_model = _hp_tune_v1(self.model, pg, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring)
                
                self.mean_cv_score = cross_val_score(self.best_model, X_tr, y_tr, cv=self.cv, scoring=self.cv_scoring).mean()
                self.best_model.fit(X_tr, y_tr)
                self.ho_score = self.ho_scoring_func(y_ho, self.best_model.predict(X_ho))
        return self
    
    
    def predict(self, X):
        return self.best_model.predict(X) 
            
class SklearnHelperMulticollinearityReducer(BaseEstimator, TransformerMixin):
    '''
    удаление мультиколлинеарности (treshold - пороговое значение)
    '''
    def __init__(self, treshold):
        self.treshold = treshold
    def fit(self, X, y=None):
        assert isinstance(X, np.ndarray)
        D_multicoll = defaultdict(list)
        for i in tqdm_notebook(range(X.shape[1])):
            for j in range(i+1, X.shape[1]):
                x1, x2 = X[:, i], X[:, j]
                corr_x1_x2 = np.abs(np.corrcoef(x1, x2)).min()
                if corr_x1_x2>=self.treshold:
                    D_multicoll[i].append(j) 
        L_multicoll = []
        for k, v in D_multicoll.items():
            v.append(k)
            L_multicoll.append(v)
        del D_multicoll
        gc.collect()
        if len(L_multicoll)>0:
            self.to_drop = []
            for idxs in L_multicoll:
                _df = pd.DataFrame(X[:, idxs]).astype(float)                
                corr_w_target_abs = _df.corrwith(pd.Series(y)).abs()
                self.to_drop.extend(corr_w_target_abs.drop(corr_w_target_abs.idxmax()).index.tolist())
            self.to_drop = np.unique(self.to_drop)
        return self
    def transform(self, X):
        try:
            return np.delete(X, self.to_drop, axis = 1)
        except:
            return X

