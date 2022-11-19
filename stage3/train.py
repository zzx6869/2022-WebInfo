import data_process
import numpy as np
import pandas as pd
import os
# 用户<1163240>没有相似的用户
# 48190738没有相似用户已移除
# 用户<2576305>没有相似的用户
# 用户<4344558>没有相似的用户
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv(r'test.csv')
newlist=[]
global tmp,k
tmp=0

# print(train_csv.head())
# print(test_csv.head())
# print(train_csv.corr())


def data_load():
    r_cols = {'user_id': np.int32,
              'movie_id': np.int32, 'movie_score': np.float32}
    ratings = pd.read_csv('train.csv', dtype=r_cols,
                          usecols=range(3), encoding="utf-8")
    ratings_matrix = ratings.pivot_table(index=['user_id'], columns=[
                                         'movie_id'], values='movie_score')
    ratings_matrix.to_csv('cache.csv')
    # print(ratings_matrix)
    return ratings_matrix


def sim_compute(ratings_matrix):
    # print(ratings_matrix)
    similarity = ratings_matrix.T.corr()
    similarity.to_csv('similar_cache.csv')
    return similarity


def predict(uid, mid, ratings_matrix, user_similar):
    global x
    similar_users = user_similar[uid].drop([uid]).dropna()
    similar_users = similar_users.where(similar_users > 0).dropna()
    if similar_users.empty is True:
        newlist.append(0)
        raise Exception("用户<%d>没有相似的用户" % uid)

    # 2.从uid用户的近邻相似用户中筛选出对iid物品有评分记录的近邻用户
    ids = set(ratings_matrix[mid].dropna().index) & set(similar_users.index)
    finally_similar_users = similar_users.loc[list(ids)]

    # 3.结合uid用户与其近邻用户的相似度预测uid用户对iid物品的评分
    sum_up = 0  # 评分预测公式的分子部分的值
    sum_down = 0  # 评分预测公式的分母部分的值
    for sim_uid, similarity in finally_similar_users.iteritems():
        # 近邻用户的评分数据
        sim_user_rated_movies = ratings_matrix.loc[sim_uid].dropna()
        # 近邻用户对iid物品的评分
        sim_user_rating_for_item = sim_user_rated_movies[mid]
        # 计算分子的值
        sum_up += similarity * sim_user_rating_for_item
        # 计算分母的值
        sum_down += similarity

    # 计算预测的评分值并返回
    predict_rating = sum_up / sum_down
    # print("预测出用户<%d>对电影<%d>的评分：%0.2f" % (uid, mid, predict_rating))
    newlist.append(round(predict_rating,2))
    return round(predict_rating, 2)

# 添加过滤规则


# def predict_all(uid, item_ids, ratings_matrix, user_similar):
#     # '''
#     # 预测全部评分
#     # :param uid:用户id
#     # :param item_ids:要预测的物品id列表
#     # :param ratings_matrix:用户-物品打分矩阵
#     # :param user_similar:用户两两间的相似度
#     # :return:生成器，逐个返回预测评分
#     # '''
#     # 逐个预测
#     for iid in item_ids:
#         try:
#             rating = predict(uid, iid, ratings_matrix, user_similar)
#         except Exception as e:
#             print(e)
#         else:
#             yield uid, iid, rating
# 添加过滤规则
def _predict_all(uid,item_ids,ratings_matrix,user_similar):
    '''
    预测全部评分
    :param uid:用户id
    :param item_ids:要预测的物品id列表
    :param ratings_matrix:用户-物品打分矩阵
    :param user_similar:用户两两间的相似度
    :return:生成器，逐个返回预测评分
    '''
    # 逐个预测
    for iid in item_ids:
        try:
            rating = predict(uid,iid,ratings_matrix,user_similar)
        except Exception as e:
            print(e)
        else:
            yield uid,iid,rating
            
def predict_all(uid,ratings_matrix,user_similar,filter_rule=None):
    '''
    预测全部评分，并可根据条件进行前置过滤
    :param uid:用户id
    :param ratings_matrix:用户-物品打分矩阵
    :param user_similar:用户两两间的相似度
    :param filter_rule:过滤规则，只能是四选一，否则将抛异常："unhot","rated",["unhot","rated"],None
    :return:生成器，逐个返回预测评分
    '''
    global k
    k=0
    global tmp
    # print(test_csv.index.size)
    for j in range(tmp,test_csv.index.size):
        if(test_csv['user_id'][j]==i):
            k+=1
        else :
            tmp=j
            break
    item_ids = test_csv['movie_id'][j-k:j]
    yield from _predict_all(uid,item_ids,ratings_matrix,user_similar)
    
def top_k_rs_result(user_id,k):
    ratings_matrix = data_load()
    user_similar = sim_compute(ratings_matrix)
    results = predict_all(user_id,ratings_matrix,user_similar)
    return sorted(results,key=lambda x: x[2],reverse=True)[:k]

if __name__ == '__main__':
    ratings_matrix = data_load()
    # user_similar = sim_compute(ratings_matrix)
    # predict_all(1, ratings_matrix.columns, ratings_matrix, user_similar)
    print(ratings_matrix)
    from pprint import pprint
    for i in ratings_matrix.index:
        result = top_k_rs_result(i,20)
    test_csv['predict_score']=newlist
    test_csv.to_csv('new.csv',index=False,sep=',')
