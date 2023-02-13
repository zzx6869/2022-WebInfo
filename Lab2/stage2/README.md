# Stage2 【3】说明

> 张展翔
>
> PB20111669

## 实验内容

### KG的构建

可以通过`rename`函数和`concat`拼接函数来实现为KG添加逆向三元组和三元组的拼接

关系数则为`kg_data`中r列的最大值再加上1

实体数为`kg_data`中h列和t列中的最大值加1

三元组的数量即为`kg_data`的长度

```python
self.n_relations = max(self.kg_data['r']) + 1
self.n_entities=max(max(self.kg_data['h']),max(kg_data['t'])) + 1
self.n_kg_data = len(self.kg_data)
```

构建字典时即只需把kg_data中特定列取出即可

```python
for row in self.kg_data.iterrows():
            h, r, t = row[1]
            self.kg_dict[h].append((t, r))
            self.relation_dict[r].append((h, t))
```

### TransE,TransR算法的实现

#### TransE

TransE即将关系视为了向量空间中头实体和尾实体之间的操作

首先，需要对关系嵌入、头实体嵌入、尾实体嵌入以及负采样的尾实体嵌入进行L2归一化处理

可以采用`torch`中的`normalize`函数实现

三元组的得分涉及向量距离的运算，具体方式如下

```python
pos_score = torch.sqrt(torch.sum(torch.pow(h_embed + r_embed - pos_t_embed,2),dim=1))                                
neg_score = torch.sqrt(torch.sum(torch.pow(h_embed + r_embed - neg_t_embed,2),dim=1))                                          
```

然后使用BPR进行优化，使负样本得分大于正样本

```python
kg_loss =torch.mean ((float)(-1) * F.logsigmoid(neg_score - pos_score))
```

#### TransR

TransR是在TransE的投影基础上将其投放到空间上，即将投影向量转为投影矩阵，因此需将h和t投影到r所在的空间中

可以利用torch中的squeeze函数进行维度运算

```python
r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)            
r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)
```

然后计算三元组的分数

```python
pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)                                   
neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)                                   
```

其余处理和TransE中大致类似

### 注入图谱实体语义信息的方式

以`calc_score`中`item_cf_embed`的计算为例

```python
item_cf_embed =  item_embed+item_kg_embed# 相加
item_cf_embed =  torch.mul(item_embed,item_kg_embed)# 相乘           
item_cf_embed=torch.concat([item_embed,item_kg_embed],dim=1)# 拼接
```

### 多任务优化与迭代优化

#### 多任务优化

多任务优化即计算出kg的损失函数和cf的损失函数，并使它们相加，然后一同进行反向传播更新等操作，即代码模板所给的原有方式

对应代码如下

```python
# train kg & cf
        time1 = time()
        total_loss = 0
        n_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.kg_dict, data.kg_batch_size, data.n_entities)

            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, is_train=True)

            if np.isnan(batch_loss.cpu().detach().numpy()):
                logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

            if (iter % args.print_every) == 0:
                logging.info('KG & CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_batch, time() - time2, batch_loss.item(), total_loss / iter))
        logging.info('KG & CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_batch, time() - time1, total_loss / n_batch))

```

及

```python
 def calc_loss(self, user_ids, item_pos_ids, item_neg_ids, h, r, pos_t, neg_t):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)

        h:              (kg_batch_size)
        r:              (kg_batch_size)
        pos_t:          (kg_batch_size)
        neg_t:          (kg_batch_size)
        """
        if self.KG_embedding_type == 'TransR':
            calc_kg_loss = self.calc_kg_loss_TransR
        elif self.KG_embedding_type == 'TransE':
            calc_kg_loss = self.calc_kg_loss_TransE
        
        kg_loss = calc_kg_loss(h, r, pos_t, neg_t)
        cf_loss = self.calc_cf_loss(user_ids, item_pos_ids, item_neg_ids)
        
        loss = kg_loss + cf_loss    #多任务优化
        return loss
```

#### 迭代优化

迭代优化则需要分开计算出kg的损失函数和cf的损失函数，分别进行反向传播更新和清梯度，二者交替进行

参照gnn中的迭代优化方式，需要更改`main_Embedding_based.py`文件中的train model部分，

将cf_loss和kg_loss分开计算

具体代码如下

```python
       # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_cf_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.kg_dict, data.kg_batch_size, data.n_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode=args.KG_embedding_type)

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))

```

相应的，在`Embedding_based.py`中将`forward`函数改为如下

```python
def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'TransR':
            return self.calc_kg_loss_TransR(*input)
        if mode == 'TransE':
            return self.calc_kg_loss_TransE(*input)
        if mode == 'predict':
            return self.calc_score(*input)
```

### 如何改进模型

参照文档所给链接，可以递归的从节点的邻居传播嵌入以细化节点的嵌入，并利用注意机制来区分邻居的重要性

例如如下函数:

```python
def _build_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embed'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embed')
            all_weights['item_embed'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embed')
            print('using xavier initialization')
        else:
            all_weights['user_embed'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                    name='user_embed', dtype=tf.float32)
            all_weights['item_embed'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                    name='item_embed', dtype=tf.float32)
            print('using pretrained initialization')

        all_weights['kg_entity_embed'] = tf.Variable(initializer([self.n_entities, 1, self.emb_dim]),
                                                     name='kg_entity_embed')
        all_weights['kg_relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]),
                                                       name='kg_relation_embed')

        all_weights['trans_W'] = tf.Variable(initializer([self.n_relations, self.emb_dim, self.kge_dim]))

        return all_weights
```

等等

## 实验结果

KG_free运行结果

```
2023-02-13 19:54:28,560 - root - INFO - Best CF Evaluation: Epoch 0040 | Precision [0.2966, 0.2532], Recall [0.0660, 0.1094], NDCG [0.3110, 0.2829]
```

多任务+相加的注入实体信息＋TransE运行结果

```
2023-02-13 20:24:53,213 - root - INFO - Best CF Evaluation: Epoch 0040 | Precision [0.2966, 0.2557], Recall [0.0653, 0.1137], NDCG [0.3084, 0.2828]
```

多任务+相加的注入实体信息＋TransR运行结果

```
2023-02-13 20:36:37,507 - root - INFO - Best CF Evaluation: Epoch 0030 | Precision [0.3020, 0.2584], Recall [0.0671, 0.1151], NDCG [0.3114, 0.2850]
```

多任务+相乘的注入实体信息＋TransE运行结果

```python
2023-02-13 20:46:12,003 - root - INFO - Best CF Evaluation: Epoch 0020 | Precision [0.2846, 0.2409], Recall [0.0655, 0.1073], NDCG [0.3010, 0.2718]
```

多任务+相乘的注入实体信息＋TransR运行结果

```python
2023-02-13 20:57:03,459 - root - INFO - Best CF Evaluation: Epoch 0020 | Precision [0.2743, 0.2336], Recall [0.0616, 0.1054], NDCG [0.2874, 0.2609]
```

多任务+拼接的注入实体信息＋TransE运行结果

```python
2023-02-13 21:08:49,934 - root - INFO - Best CF Evaluation: Epoch 0040 | Precision [0.2966, 0.2555], Recall [0.0653, 0.1136], NDCG [0.3085, 0.2827]
```

多任务+拼接的注入实体信息＋TransR运行结果

```python
2023-02-13 21:20:11,018 - root - INFO - Best CF Evaluation: Epoch 0030 | Precision [0.3020, 0.2584], Recall [0.0671, 0.1151], NDCG [0.3117, 0.2851]
```



迭代＋相加的注入实体信息＋TransE运行结果

```python
2023-02-13 20:53:31,142 - root - INFO - Best CF Evaluation: Epoch 0020 | Precision [0.2913, 0.2492], Recall [0.0662, 0.1076], NDCG [0.3059, 0.2778]
```

迭代＋相加的注入实体信息＋TransR运行结果

```python
2023-02-13 21:14:05,894 - root - INFO - Best CF Evaluation: Epoch 0040 | Precision [0.2890, 0.2510], Recall [0.0669, 0.1109], NDCG [0.3007, 0.2775]
```

迭代＋相乘的注入实体信息＋TransE运行结果

```python
2023-02-13 21:06:13,109 - root - INFO - Best CF Evaluation: Epoch 0030 | Precision [0.2864, 0.2597], Recall [0.0638, 0.1119], NDCG [0.3010, 0.2837]
```

迭代＋相乘的注入实体信息＋TransR运行结果

```python
2023-02-13 21:30:08,820 - root - INFO - Best CF Evaluation: Epoch 0040 | Precision [0.2877, 0.2530], Recall [0.0644, 0.1099], NDCG [0.3022, 0.2802]
```

迭代＋拼接的注入实体信息＋TransE运行结果

```python
2023-02-13 20:59:42,602 - root - INFO - Best CF Evaluation: Epoch 0020 | Precision [0.2913, 0.2492], Recall [0.0662, 0.1076], NDCG [0.3059, 0.2778]
```

迭代＋拼接的注入实体信息＋TransR运行结果

```python
2023-02-13 21:22:09,666 - root - INFO - Best CF Evaluation: Epoch 0040 | Precision [0.2890, 0.2506], Recall [0.0669, 0.1106], NDCG [0.3007, 0.2772]
```



|                                   | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --------------------------------- | -------- | --------- | ------ | ------- |
| MF                                | 0.0660   | 0.1094    | 0.3110 | 0.2829  |
| 多任务+相加的注入实体信息＋TransE | 0.0653   | 0.1137    | 0.3084 | 0.2828  |
| 多任务+相加的注入实体信息＋TransR | 0.0671   | 0.1151    | 0.3114 | 0.2850  |
| 多任务+相乘的注入实体信息＋TransE | 0.0655   | 0.1073    | 0.3010 | 0.2718  |
| 多任务+相乘的注入实体信息＋TransR | 0.0616   | 0.1054    | 0.2874 | 0.2609  |
| 多任务+拼接的注入实体信息＋TransE | 0.0653   | 0.1136    | 0.3085 | 0.2827  |
| 多任务+拼接的注入实体信息＋TransR | 0.0671   | 0.1151    | 0.3117 | 0.2851  |
| 迭代＋相加的注入实体信息＋TransE  | 0.0662   | 0.1076    | 0.3059 | 0.2778  |
| 迭代＋相加的注入实体信息＋TransR  | 0.0669   | 0.1109    | 0.3007 | 0.2775  |
| 迭代＋相乘的注入实体信息＋TransE  | 0.0638   | 0.1119    | 0.3010 | 0.2837  |
| 迭代＋相乘的注入实体信息＋TransR  | 0.0644   | 0.1099    | 0.3022 | 0.2802  |
| 迭代＋拼接的注入实体信息＋TransE  | 0.0662   | 0.1076    | 0.3059 | 0.2778  |
| 迭代＋拼接的注入实体信息＋TransR  | 0.0669   | 0.1106    | 0.3007 | 0.2772  |



