import pandas as pd
import gplearn.genetic
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau, pearsonr, spearmanr

def kendall_pval(x,y):
    return kendalltau(x,y)[1]

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]

def spearmanr_pval(x,y):
    return spearmanr(x,y)[1]

def symbolic_transform(train_features, train_labels, test_features, test_labels,\
    function_set=('add', 'sub', 'mul', 'div', 'abs', 'neg', 'inv', 'max', 'min' ), 
    parsimony_coefficient=0.01, **kargs):
    '''
    train_labels, test_labels should be numeric
    Return a dict containing symbolic transformer, train and test correlations
    '''
    dict_to_return = {}

    c = gplearn.genetic.SymbolicTransformer(
        function_set=function_set, parsimony_coefficient=parsimony_coefficient, **kargs)

    c.fit(train_features, train_labels)
    dict_to_return['SymbolicTransformer'] = c

    trans = c.transform(train_features)
    df_trans = pd.concat([pd.DataFrame(trans), train_labels.reset_index()], axis=1)
    dict_to_return['train_correlation'] = df_trans.corr()
    dict_to_return['train_correlation_pvalue'] = df_trans.corr(method=pearsonr_pval)
    

    trans = c.transform(test_features)
    df_trans = pd.concat([pd.DataFrame(trans), test_labels.reset_index()], axis=1)
    dict_to_return['test_correlation'] = df_trans.corr()
    dict_to_return['test_correlation_pvalue'] = df_trans.corr(method=pearsonr_pval)
    
    return dict_to_return

class SymbolicTransformerLikeSklearn():
    def __init__(self):
        self.transformer = gplearn.genetic.SymbolicTransformer(n_components=1)
    
    def fit(self, X, y):
        self.transformer.fit(train_features, train_labels)

    def predict_proba(self, X):
        y = self.transformer.transform(X).flatten()
        return y
    
    def get_params(self, **kargs):
        return self.transformer.get_params(self)