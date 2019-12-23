import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization

train = pd.read_csv('train.csv')

x_train = train.drop(columns = 'trip_duration')
y_train = train['trip_duration']

dm_train = xgb.DMatrix(x_train, label = list(y_train))


def run_xgb_CV(eta, max_depth, gamma, colsample_bytree, min_child_weight):
    params = {'eval_metric': 'mae',
              'booster': 'gbtree',
              'eta': eta,
              'max_depth': int(max_depth),
              'subsample': 0.7,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree, 
              'min_child_weight': min_child_weight}
    cv_result = xgb.cv(params, dm_train, num_boost_round = 100, nfold = 5)    
    
    # Bayesian optimisation library can only maximise, so to minimise, the negative MAE is returned
    return -1.0 * cv_result['test-mae-mean'].iloc[-1]


xgb_optimiser = BayesianOptimization(run_xgb_CV, {'eta': (0.05, 0.3),
                                             	  'max_depth': (5, 30), 
                                             	  'gamma': (0.2, 0.7),
                                             	  'colsample_bytree': (0.3, 0.8),
                                             	  'min_child_weight': (5, 30)})

xgb_optimiser.maximize(init_points = 20, n_iter = 30)

params = xgb_optimiser.max['params']
params['max_depth'] = int(params['max_depth'])


param_df = pd.DataFrame.from_dict(params, orient='index').reset_index()
param_df.columns = ['parameter_name', 'value']
param_df.to_csv('xgb_parameters.csv', index = False)