import pandas as pd
import numpy as np
import pickle

from google.cloud import bigquery

from cluster_feature_functions import find_allfeatures
from growth_forecasting_feature_functions import growth_ratings

from tqdm import tqdm
def cluster_features(FY, client):
    print(f"\nFinding {FY} Features")
    return find_allfeatures(FY, client)

def cluster_classification(FY, client):
    print(f"\nFinding {FY} Predictions")
    return growth_ratings(FY, client)

client = bigquery.Client()
cluster_data = pd.concat([cluster_features(FY,client) for FY in range(2015,2025,1)])
with open('../data/all_cluster_data.pckl','wb') as fil:
    pickle.dump(cluster_data, fil)

cluster_prediction_data = pd.concat([cluster_classification(FY,client) for FY in range(2015,2025,1)])
with open('../data/all_cluster_predictions.pckl','wb') as fil:
    pickle.dump(cluster_prediction_data, fil)
