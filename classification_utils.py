
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, auc, roc_curve

from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from plot_utils import plot_roc

import matplotlib.pyplot as plt


def GridSearchCVForMultipleAlgs(
        X, y,
        estimators = {
            'RandomForest': RandomForestClassifier(),
            'SVC': SVC(),
            'ExtraTrees': ExtraTreesClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'GradientBoosting': GradientBoostingClassifier(),
            'MLP': MLPClassifier(),
            'LogisticRegression': LogisticRegression(),
            'DecisionTree': DecisionTreeClassifier(),
            'KNN': KNeighborsClassifier()
        },
        param_grids = {
            'RandomForest': {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]},
            'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'ExtraTrees': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
            'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]},
            'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
            'MLP': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['tanh', 'relu']},
            'LogisticRegression': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']},
            'DecisionTree': {'max_depth': [None, 10, 20], 'criterion': ['gini', 'entropy']},
            'KNN': {'n_neighbors': [3, 5, 7], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
        }
    ):
    results = {}

    for name, estimator in estimators.items():
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grids[name], scoring='accuracy', cv=5, verbose=2)
        grid_search.fit(X, y)
        results[name] = {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

    # 输出每个估计器的最佳结果
    for name, result in results.items():
        print(f"{name} - Best Accuracy: {result['best_score']} with params: {result['best_params']}")

    return results

def cross_val_predict_multiple_algs(
    X,
    y,
    cv=5,
    estimators=[ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), MLPClassifier()],
    manual_indicator=None,
    manual_indicator_name=None,
    plot=True,
    plot_dir='./cross_val_predict_multiple_algs/'):
    # label_col = 'her2_status_by_ihc'

    # estimator.fit(
    #     df_density_per_tissue_all_samples_pivot_table_append_tissue_percentage_append_ihc_pr_0_1[all_possible_input_features], 
    #     df_density_per_tissue_all_samples_pivot_table_append_tissue_percentage_append_ihc_pr_0_1['pr_status_by_ihc']
    # )

    all_fpr = dict()
    all_tpr = dict()
    all_auc = dict()
    for estimator in estimators:        
        print(estimator)

        predictions_proba = cross_val_predict(estimator, 
                                            X, # df_density_per_tissue_all_samples_pivot_table_append_tissue_percentage_append_ihc_pr_0_1[feature_cols_from_large_areas], 
                                            y, # df_density_per_tissue_all_samples_pivot_table_append_tissue_percentage_append_ihc_pr_0_1[label_col].replace({'Positive':1, 'Negative': 0,'Indeterminate':0,'Equivocal':0,'[Not Evaluated]':0,'[Not Available]':0}),
                                            cv=cv,
                                            method='predict_proba')

        fpr, tpr, _ = roc_curve(y, predictions_proba[:, 1])
        auc_score = auc(fpr, tpr)
        
        # 存储结果
        all_fpr[estimator.__class__.__name__] = fpr
        all_tpr[estimator.__class__.__name__] = tpr
        all_auc[estimator.__class__.__name__] = auc_score


        if plot:
            # This is only for 2-class problem.
            os.makedirs(plot_dir, exist_ok=True)
            path_output = os.path.join(plot_dir, '{}.pdf'.format(estimator))
            plot_roc(
                y,
                predictions_proba[:,1]
            )
            plt.savefig(path_output)
            # plt.show()

    if manual_indicator is not None:
        
        fpr, tpr, _ = roc_curve(y, manual_indicator)
        auc_score = auc(fpr, tpr)
        
        # 存储结果
        all_fpr[manual_indicator_name] = fpr
        all_tpr[manual_indicator_name] = tpr
        all_auc[manual_indicator_name] = auc_score


    if plot:

        plt.figure()
        
        # 为每个分类器绘制ROC曲线
        for alg_name, auc_score in all_auc.items():
            plt.plot(all_fpr[alg_name], all_tpr[alg_name],
                    label='%s (AUC = %0.2f)' % (alg_name, auc_score))
        
        # 设置图的属性
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # 保存图形
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'roc_comparison.pdf'))
        plt.close() 

if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10)
    GridSearchCVForMultipleAlgs(X, y)