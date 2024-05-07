import pandas as pd
import joblib

import classification_utils

from classification_utils import GridSearchCVForMultipleAlgs


df_density_per_tissue_all_samples_pivot_table_append_tissue_percentage_append_ihc = pd.read_csv('df_density_per_tissue_all_samples_pivot_table_append_tissue_percentage_append_ihc_gytm.csv', index_col=0)
all_possible_labels = joblib.load('all_possible_labels_gytm.pkl')
all_possible_input_features = joblib.load('all_possible_input_features_gytm.pkl')

prediction_results = {}

for y_label in all_possible_labels:
    X = df_density_per_tissue_all_samples_pivot_table_append_tissue_percentage_append_ihc[all_possible_input_features].fillna(0).values
    y = df_density_per_tissue_all_samples_pivot_table_append_tissue_percentage_append_ihc[y_label].fillna('nan').values
    prediction_result = GridSearchCVForMultipleAlgs(X, y)
    prediction_results[y_label] = prediction_result

joblib.dump(prediction_results, 'prediction_results_gytm.pkl')