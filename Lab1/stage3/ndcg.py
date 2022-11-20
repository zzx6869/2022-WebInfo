from sklearn.metrics import ndcg_score
import pandas as pd
import numpy as np
# ndcg=0.9814199692967567
# ndcg@5=0.9815151048429968
# ndcg@10=0.9421806099621497
file='all.csv'
true_score=[pd.read_csv(file)['movie_score'].values]
predict_score=[pd.read_csv(file)['predict_score'].values]
ndcg=ndcg_score(true_score, predict_score,k=10)
print(ndcg)