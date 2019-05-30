import time
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define default parameters 
# params = {
#     'max_depth': 6,
#     'min_child_weight':1,
#     'subsample':1,
#     'colsample_bytree':1,
#     'eta':.3,
#     'objective':'binary:logistic',
#     'eval_metrics': "rmse"
#         }
# early_stopping_round = 10
# evals = [(dtest, "Test")]
# num_boost_round = 999

class XGBoostOptimizer1:
    def __init__(self
                 ,dtrain = None
                 ,dtest = None
                 ,params = {
                            'max_depth': 6,
                            'min_child_weight':1,
                            'subsample':1,
                            'colsample_bytree':1,
                            'eta':.3,
                            'objective':'binary:logistic',
                            'eval_metrics': "rmse"
                                }
				 ,cv_metrics = "rmse"
                 ,early_stopping_round = 10
                 ,num_boost_round = 999
                 ,seed=42
                 ,nfold=5
                ):
        self.stage_metric = []
        self.dtrain = dtrain
        self.dtest = dtest
        self.params = params
        self.early_stopping_round = early_stopping_round
        self.num_boost_round = num_boost_round
        self.seed = seed
        self.nfold = nfold
        self.cv_metrics = cv_metrics
        self.stages = ['complexity','feature-samp','learning-rate']
        self.final_model = None

    def optimize_tree(self,level='complexity'):
        print(f"Level selected = {level}")
        if level =='complexity':
            #Tuning the complexity of the tree. 
            #max_depth and min_child_weight should be tuned together
            gridsearch_params = [
                                    (max_depth, min_child_weight)
                                    for max_depth in range(9,12)
                                    for min_child_weight in range(5,8)
                                ]
            param_to_opt = ['max_depth','min_child_weight']
        elif level == 'feature-samp':
            gridsearch_params = [
                                (subsample, colsample)
                                for subsample in [i/10. for i in range(7,11)]
                                for colsample in [i/10. for i in range(7,11)]
                            ][::-1]
            
            param_to_opt = ['subsample','colsample_bytree']
        elif level =='learning-rate':
            gridsearch_params = [
                                (eta, -1)
                                for eta in [.3, .2, .1, .05, .01, .005]
                                ]
            param_to_opt = ['eta','None']
        else:
            raise Exception("Wrong paramaters")
        
        #define initial values
        if self.cv_metrics == 'auc':
            best_metric = 0
        else:
            best_metric = float("Inf")
        best_params = None
        

        cv_params = self.params
        print('Solving best parameters ...')
        time.sleep(1)
        for param0, param1 in tqdm(gridsearch_params):
            print(f"CV with {param_to_opt[0]}={param0}, {param_to_opt[1]}={param1}")
            # Update our parameters
            cv_params[param_to_opt[0]] = param0
            if 'eta' not in param_to_opt:
                cv_params[param_to_opt[1]] = param1

            # Run CV
            cv_results = xgb.cv(
                params = cv_params,
                dtrain = self.dtrain,
                num_boost_round=self.num_boost_round,
                seed=self.seed,
                nfold=self.nfold,
                metrics=self.cv_metrics,
                early_stopping_rounds=self.early_stopping_round
            )
            # Update best MAE
            if self.cv_metrics == 'auc':
                mean_metric = cv_results[f'test-{self.cv_metrics}-mean'].max()
                boost_rounds = cv_results[f'test-{self.cv_metrics}-mean'].argmax()
                if mean_metric > best_metric:
                    best_metric = mean_metric
                    best_params = (param0,param1)
                    
            else:
                mean_metric = cv_results[f'test-{self.cv_metrics}-mean'].min()
                boost_rounds = cv_results[f'test-{self.cv_metrics}-mean'].argmin()
                if mean_metric < best_metric:
                    best_metric = mean_metric
                    best_params = (param0,param1)
            
            print("\t{} {} for {} rounds".format(self.cv_metrics,mean_metric, boost_rounds))
            self.stage_metric.append(mean_metric)

        print("Best stage params: {}, {}, {}: {}".format(best_params[0], best_params[1],self.cv_metrics, best_metric))

        self.params[param_to_opt[0]] = best_params[0]
        if 'eta' not in param_to_opt:
            self.params[param_to_opt[1]] = best_params[1]
            
    def plot_optimization(self):
        if len(self.stage_metric)>1:
            x = list(range(0,len(self.stage_metric)))
            plt.plot(x,self.stage_metric)
            plt.xlabel('Epoch')
            plt.ylabel(self.cv_metrics)
            plt.title("Optimization Trace")
            plt.show()
        else:
            print("Nothing to plot")
    def run_optimizer(self,refit=False):
        self.stage_metric = []
        for stage in self.stages:
            self.optimize_tree(level=stage)
        print("Finished optimization.\nBest params:",self.params)
        
        if refit:
            self.final_model = xgb.train(
                params=self.params,
                dtrain=self.dtrain,
                num_boost_round=self.num_boost_round,
                evals = [(self.dtest, "Test")],
                early_stopping_rounds=self.early_stopping_round
            )
    def get_best_model(self):
        return self.final_model