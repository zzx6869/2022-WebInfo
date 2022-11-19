# stage3 说明

> by 张展翔

## 文件结构



![QQ截图20221119152317](/pic/QQ截图20221119152317.png)

- Movie_score.csv为提供电影评分样本

- all.csv为最后预测评分与实际评分汇总

- cache.csv为用户-电影评分矩阵
- data_process.py为数据处理代码
- forall为总的评分样本汇总文件（未使用）
- ndcg.py为ndcg计算代码
- pic为截图保存文件夹
- similar_cache.csv为用户相关度矩阵
- test.csv和train为测试集和训练集
- train.py为预测评分代码

## 思路分析

本次预测采用基于用户相似度的预测方法，采用均值中心化来进行计算
$$
pred(u,i)=\hat{r_{ui}}=\bar{r_u}+\frac{\Sigma_{v\in U}sim(u,v)*(r_{vi}-\bar{r_v})}{\Sigma_{v\in U}{|sim(u,v)|}}
$$
其中，用户相似度sim根据皮尔逊相关系数来计算

最后的ndcg采用sklearn中的ndcg_score函数计算

ndcg=0.9814199692967567

ndcg@5=0.9815151048429968

ndcg@10=0.9421806099621497
