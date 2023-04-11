# Web Lab2实验报告

> 刘阳 PB20111677
>
> 张展翔 PB20111669 
>
> 黄鑫 PB20061174

## Stage 1图谱抽取

> 刘阳 PB20111677

图谱的抽取共使用了2个函数，函数jump接受输入对象集合items，在data_base中寻找首对象或者尾对象在集合items中的三元组，保存在文件中，并返回新的对象集合。函数opt则对得到的三元组集合进行优化，根据输入的数量筛选掉数量较少的对象及关系，得到优化后的知识图谱：

```python
def jump(items: set, out_file):
    # cnt = 0
    new_items = set(items)
    with gzip.open('../../../freebase_douban.gz') as freebase_file:
        for line in freebase_file:
            triplet = line.strip().decode().split('\t')
            triplet = triplet[0:3]
            for i in range(0, len(triplet)):
                triplet[i] = re.findall('(?<=<http://rdf.freebase.com/ns/).*(?=>)', triplet[i])
                if (triplet[i] == []):
                    break
                triplet[i] = triplet[i][0]
            else:
                if (triplet[0] in items or triplet[2] in items):
                    new_items.add(triplet[2])
                    new_items.add(triplet[0])
                    out_file.write(line)
                    # print(line)
    return new_items
```

```python
def opt(in_file, out_file, triplet_num, obj_num):
    items = dict()
    relations = dict()
    for line in in_file:
        triplet = line.strip().decode().split('\t')
        triplet = triplet[0:3]
        for i in range(0, len(triplet)):
            triplet[i] = re.findall('(?<=<http://rdf.freebase.com/ns/).*(?=>)', triplet[i])
            triplet[i] = triplet[i][0]
        if (items.get(triplet[0])):
            items[triplet[0]] += 1
        else:
            items[triplet[0]] = 1
        if (items.get(triplet[2])):
            items[triplet[2]] += 1
        else:
            items[triplet[2]] = 1
        if (relations.get(triplet[1])):
            relations[triplet[1]] += 1
        else:
            relations[triplet[1]] = 1
    # with open('./tmp.json', 'wb') as tmp:
    #     tmp.write(json.dumps([items, relations]).encode())
    in_file.seek(0, 0)
    for line in in_file:
        triplet = line.strip().decode().split('\t')
        triplet = triplet[0:3]
        for i in range(0, len(triplet)):
            triplet[i] = re.findall('(?<=<http://rdf.freebase.com/ns/).*(?=>)', triplet[i])
            triplet[i] = triplet[i][0]
        if (items[triplet[0]] >= obj_num and items[triplet[2]] >= obj_num and relations[triplet[1]] >= triplet_num):
            out_file.write(line)
    return set([item for item in items if items[item] >= obj_num])
```



## Stage 2图谱推荐

>刘阳 PB20111677 【1】
>
>张展翔 PB20111669 【2】【3】【5】
>
>黄鑫 PB20061174     【2】【4】【5】

### 知识图谱映射

只需要根据stage1得到的2跳子图，统计所有出现的对象和关系构成一个map，然后保存替换即可。

```python
def main(base_file):
    obj_file = open('./obj.json', 'w')
    rela_file = open('./rela_file.json', 'w')

    objs = dict()
    obj_cnt = 0
    relas = dict()
    rela_cnt = 0

    id2proj = dict()
    with open('../stage1/douban2fb.txt', 'rb') as douban2fb_file:
        for line in douban2fb_file:
            data = line.strip().decode().split('\t')
            id2proj[data[0]] = data[1]
    items = set()
    with open('../../Lab1/stage1/Movie_id.txt', 'rb') as id_file:
        for line in id_file:
            if(id2proj.get(line.strip().decode(), None)):
                items.add(id2proj.get(line.strip().decode()))

    for item in items:
        objs[item] = obj_cnt
        obj_cnt += 1
    
    for line in base_file.readlines():
        triplet = line.strip().decode().split('\t')
        triplet = triplet[0:3]
        for i in range(0, len(triplet)):
            triplet[i] = re.findall('(?<=<http://rdf.freebase.com/ns/).*(?=>)', triplet[i])
            triplet[i] = triplet[i][0]
        if objs.get(triplet[0], None) == None:
            objs[triplet[0]] = obj_cnt
            obj_cnt += 1
        if relas.get(triplet[1], None) == None:
            relas[triplet[1]] = rela_cnt
            rela_cnt += 1
        if objs.get(triplet[2], None) == None:
            objs[triplet[2]] = obj_cnt
            obj_cnt += 1
    
    obj_file.write(json.dumps(objs))
    rela_file.write(json.dumps(relas))

    base_file.seek(0, 0)

    kg_final = open('./data/Douban/kg_final.txt', 'w')
    for line in base_file.readlines():
        triplet = line.strip().decode().split('\t')
        triplet = triplet[0:3]
        for i in range(0, len(triplet)):
            triplet[i] = re.findall('(?<=<http://rdf.freebase.com/ns/).*(?=>)', triplet[i])
            triplet[i] = triplet[i][0]
        kg_final.write(objs[triplet[0]].__str__() + ' ' + relas[triplet[1]].__str__() + ' ' + objs[triplet[2]].__str__() + '\n')

```

### 基于图谱嵌入的模型【3】

> 张展翔
>
> PB20111669

#### 实验内容

##### KG的构建

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

##### TransE,TransR算法的实现

###### TransE

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

###### TransR

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

##### 注入图谱实体语义信息的方式

以`calc_score`中`item_cf_embed`的计算为例

```python
item_cf_embed =  item_embed+item_kg_embed## 相加
item_cf_embed =  torch.mul(item_embed,item_kg_embed)## 相乘           
item_cf_embed=torch.concat([item_embed,item_kg_embed],dim=1)## 拼接
```

##### 多任务优化与迭代优化

###### 多任务优化

多任务优化即计算出kg的损失函数和cf的损失函数，并使它们相加，然后一同进行反向传播更新等操作，即代码模板所给的原有方式

对应代码如下

```python
## train kg & cf
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

###### 迭代优化

迭代优化则需要分开计算出kg的损失函数和cf的损失函数，分别进行反向传播更新和清梯度，二者交替进行

参照gnn中的迭代优化方式，需要更改`main_Embedding_based.py`文件中的train model部分，

将cf_loss和kg_loss分开计算

具体代码如下

```python
       ## train cf
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

        ## train kg
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

##### 如何改进模型

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

#### 实验结果

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

### 基于GNN的知识感知推荐【4】

> 黄鑫
>
> PB20061174

#### 实验过程([4]a-c)

按照要求补全实验数据加载、处理，以及图卷积相关操作的代码

##### loader_GNN_based补全([4].a)

###### 首先为映射好的知识图谱三元组添加逆向三元组,随后计算相关数据

```python
        ## 1. 为KG添加逆向三元组，即对于KG中任意三元组(h, r, t)，添加逆向三元组 (t, r+n_relations, h)，
        ## 并将原三元组和逆向三元组拼接为新的DataFrame，保存在 kg_data 中。
        n_relations = max(kg_data['r']) + 1
        inverse_triplets = kg_data.copy()
        inverse_triplets = inverse_triplets.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_triplets['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_triplets], ignore_index=True)
        
        ## TODO[done]: 2. 计算关系数，实体数，实体和用户的总数
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']),max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities
   
```

###### 更改训练数据和字典中的索引值

```python
        
        #TODO[done]: 3. 使用 map()函数 将 self.cf_train_data 和 self.cf_test_data 中的 用户索引 范围从[0, num of users)
        ##    映射到[num of entities, num of entities + num of users)，并保持原有数据形式和结构不变
        self.cf_train_data = (list(map(lambda x: x + self.n_entities, self.cf_train_data[0])), self.cf_train_data[1])
        self.cf_test_data = (list(map(lambda x: x + self.n_entities, self.cf_test_data[0])), self.cf_test_data[1])

        #TODO[done]: 4. 将 self.train_user_dict 和 self.test_user_dict 中的用户索引（即key值）范围从[0, num of users)
        ##    映射到[num of entities, num of entities + num of users)，并保持原有数据形式和结构不变
        user_train = list(self.train_user_dict.keys())
        for user_idx in user_train:
            self.train_user_dict[user_idx + self.n_entities] = self.train_user_dict.pop(user_idx)
        user_test = list(self.test_user_dict.keys())
        for user_idx in user_test:
            self.test_user_dict[user_idx + self.n_entities] = self.test_user_dict.pop(user_idx)
```

###### 重构交互数据和逆向交互数据

```python
 #TODO[done]: 5. 以三元组的形式 (user, 0, movie) 重构交互数据，其中 关系0 代表 like
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = pd.Series(list(self.cf_train_data[0]))
        cf2kg_train_data['t'] = pd.Series(self.cf_train_data[1])
        #print(cf2kg_train_data)

        #TODO[done]: 6. 以三元组的形式 (movie, 1, user) 重构逆向的交互数据，其中 关系1 代表 like_by
        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = pd.Series(self.cf_train_data[1])
        inverse_cf2kg_train_data['t'] = pd.Series(list(self.cf_train_data[0]))
        #print(inverse_cf2kg_train_data)
```

###### 最后重构字典

```python
        #TODO[done]: 7. 根据 self.kg_train_data 构建字典 self.train_kg_dict ，其中key为h, value为tuple(t, r)，
        ##    和字典 self.train_relation_dict, 其中key为r，value为tuple(h, t)。
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)
        for _, row in tqdm(self.kg_train_data.iterrows(), total=self.kg_train_data.shape[0], desc='generating new dict'):
            self.train_kg_dict[row['h']] += [(row['t'], row['r'])]
            self.train_relation_dict[row['r']] += [(row['h'], row['t'])]
```

###### 实现随机游走对称归一化拉普拉斯矩阵的计算

随机游走归一化Laplace矩阵如下:
$$
L^{random \ walk} = D^{-1}A
$$

```python
            #TODO[done]: 8. 根据对称归一化拉普拉斯矩阵的计算代码，补全随机游走归一化拉普拉斯矩阵的计算代码
            rowsum = np.array(adj.sum(axis=1))
            d_inv = np.power(rowsum, -1.).flatten()
            d_mat_inv = sp.diags(d_inv) ## D^{-1}
            norm_adj = d_mat_inv.dot(adj) ## D^{-1} \dot A
            return norm_adj.tocoo()
```

##### GNN_based 补全([4].b)

###### 如图，首先获得一跳邻域表征，再按照不同的图聚合方式将中心节点表征和一条邻域融合.

```python
def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, embed_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        ## 1.TODO[done]: Equation (3) 得到一跳邻域的表征 side_embeddings
    side_embeddings = torch.matmul(A_in, ego_embeddings)                    
    if self.aggregator_type == 'gcn':
        ## 2.TODO[done]: Equation (6) 将中心节点表征和一跳邻域表征相加，再进行线性变换和非线性激活
        embeddings = ego_embeddings + side_embeddings                                                       
        embeddings = self.activation(self.linear(embeddings))                                              
    elif self.aggregator_type == 'graphsage':
        ## 3.TODO[done]: Equation (7) 将中心节点表征和一跳邻域表征拼接，再进行线性变换和非线性激活
        embeddings = torch.concat([ego_embeddings, side_embeddings], dim=1)                               
        embeddings = self.activation(self.linear(embeddings))                                               
    elif self.aggregator_type == 'lightgcn':
        ## 4.TODO[done]: Equation (8) 简单地将中心节点表征和一跳邻域表征相加
        embeddings = ego_embeddings + side_embeddings
```

###### 迭代计算实体嵌入，通过`F.normalize`进行归一化(L2范数),最后append

```python
## 5.TODO[done]: 迭代地计算每一层卷积层的实体（包含用户）嵌入，将其L2范数归一化后，append到all_embed中
for idx, layer in enumerate(self.aggregator_layers):
    ego_embed = layer(ego_embed, self.A_in)
    norm_embed = F.normalize(ego_embed, p=2, dim=1)
    all_embed.append(norm_embed)                                                                  ## (n_users + n_entities, embed_dim)
```

###### TransE算法

<img src="/pic/TransE.png" alt="image-20230214110024482" style="zoom:50%;" />

TransE算法对关系嵌入，头实体嵌入，尾实体嵌入，负采样的尾实体嵌入进行L2范数归一化后，以L2范数计算正负样本三元组的得分,最后通过BPR Loss进行优化

```python
        #TODO[done]: 11. 对关系嵌入，头实体嵌入，尾实体嵌入，负采样的尾实体嵌入进行L2范数归一化
        r_embed = F.normalize(r_embed, p=2, dim=1)
        h_embed = F.normalize(h_embed, p=2, dim=1)
        pos_t_embed = F.normalize(pos_t_embed, p=2, dim=1)
        neg_t_embed = F.normalize(neg_t_embed, p=2, dim=1)

        ## 取L2范数
        #TODO[done]: 12. 分别计算正样本三元组 (h_embed, r_embed, pos_t_embed) 和负样本三元组 (h_embed, r_embed, neg_t_embed) 的得分
        pos_score = torch.sqrt(torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1))  ## (kg_batch_size)
        neg_score = torch.sqrt(torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1))  ## (kg_batch_size)

        #TODO[done]: 13. 使用 BPR Loss 进行优化，尽可能使负样本的得分大于正样本的得分
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
```

###### TransR

<img src="/pic/TransR.png" alt="image-20230214110308880" style="zoom:50%;" />

与TranE不同，TranR首先计算实体在关系空间中的投影嵌入，使用投影嵌入进行评分,且正负样本的评分函数选择L2范数的平方,依然采用BPR Loss进行优化

```python
#TODO[done]: 7. 计算头实体，尾实体和负采样的尾实体在对应关系空间中的投影嵌入
"""
h_r = h * W_r
t_r = t * W_r
"""
r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)          ## (kg_batch_size, relation_dim)
r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  ## (kg_batch_size, relation_dim)
r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  ## (kg_batch_size, relation_dim)

#TODO[done]: 8. 对关系嵌入，头实体嵌入，尾实体嵌入，负采样的尾实体嵌入进行L2范数归一化
r_embed = F.normalize(r_embed, p=2, dim=1)
r_mul_h = F.normalize(r_mul_h, p=2, dim=1)
r_mul_pos_t = F.normalize(r_mul_pos_t, p=2, dim=1)
r_mul_neg_t = F.normalize(r_mul_neg_t, p=2, dim=1)
## || h_r + r - t_r ||^2 :取L2范数的平方
#TODO[done]: 9. 分别计算正样本三元组 (h_embed, r_embed, pos_t_embed) 和负样本三元组 (h_embed, r_embed, neg_t_embed) 的得分
pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     ## (kg_batch_size)
neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     ## (kg_batch_size)

## Equation (2)
#TODO[done]: 10. 使用 BPR Loss 进行优化，尽可能使负样本的得分大于正样本的得分
kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
kg_loss = torch.mean(kg_loss)
```

##### 多任务优化([4].c)

原代码框架使用迭代更新的方式(先计算CF Loss，在反向传播更新权重后，再计算KG Loss，基于KG Loss反向传播更新权重,按照此方式往复循环)

要修改成多任务方式，只需要在一次迭代内同时计算出CF和KG的Loss,令$Loss = Loss_{CF} + \lambda Loss_{KG}$(默认取$\lambda=1$)使用$Loss$反向传播，再更新权重

```python
## 多任务
                kg_cf_batch_loss = cf_batch_loss + args.multitasks_lambda * kg_batch_loss
                kg_cf_batch_loss.backward()
                kg_cf_optimizer.step()
                kg_cf_optimizer.zero_grad()
                kg_cf_total_loss += kg_cf_batch_loss.item()
```

#### 实验结果

##### 结果分析

由于不同的图谱嵌入方法、不同的训练方式、不同的图卷积聚合方式以及图卷积层的数量等对最终训练效果都有影响，因此要考虑到所有情况则需要训练很多次。由于缺少硬件条件，得到的实验结果仅能对个别方面进行对比分析探讨

训练方式为KG,CF迭代训练,图卷积指定使用归一化Laplace矩阵,我们每10个epoch就对训练模型进行一次评估.

当最佳评估指标与当前评估在评估List中位置相差大于10时,我们认为模型已经收敛到合适程度,停止训练.

选择图聚合为`lightgcn`,`gcn`,`graphsage`和图嵌入为`TransE`,`TranR`,组合下总共六种情况

以下为不同情况下迭代次数和训练指标(Recall@K, NDCG@K, K=5,10)的曲线图.

粉红色曲线为KG_free_based的训练指标

![Figure_1](/pic/Figure_1.png)

![Figure_2](/pic/Figure_2.png)

通过比对曲线图我们不难发现:

####### 1.相同拉普拉斯矩阵和聚合方式下,TransE效果普遍比TranR差,同时采用TransR进行图嵌入能得到明显的提升:

从曲线图可以发现，采用TranE嵌入不仅迭代的次数更多，且效果都比TransR和不使用KG的方法差.采用TransR嵌入的评估指标普遍好于不引入KG的方法

####### 2.相同拉普拉斯矩阵下,采用TransE嵌入时,效果上:gcn>graphsage>lightgcn;采用TransR嵌入时,效果上:lightgcn>graphsage>gcn:

从曲线图可以发现,TransE下采用gcn最优,TransR下lightgcn反而最优

####### 3.随机游走矩阵可能对模型有提升

我们将拉普拉斯矩阵换成随机游走形式时(见图中最高的灰色曲线)，在TransR和lightgcn下，指标与归一化拉普拉斯矩阵相比进一步提升(由于没有好的硬件条件和时间限制，暂时无法完成更多其他条件下的训练以对比)

####### 4.与TransE相比,采用TransR模型能够更快地收敛

从曲线图中可见,采用TransR算法收敛地更快(在epoch轴方向上延伸更短，证明训练停止地更快)

#### 模型改进([4].d)

###### 新的聚合方式:Bi-Interaction Aggregator

采用如下的新的聚合方式

$f_{Bi−Interaction}=LeakyReLU(W_1(e_h+e_{Nh}))+LeakyReLU(W_2(e_h⊙e_{Nh}))$

- ⊙ 对应元素积，让相似的实体传递更多的信息

其中$e_h$表示实体嵌入表示,而$e_{N_h}$为实体的领域嵌入表示，采用该聚合方式能够考虑两者的特征交互.代码实现如下

```python
elif self.aggregator_type == 'bi-interaction':
     sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
     bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
     embeddings = bi_embeddings + sum_embeddings
```

由于加入了**feature-interaction**，对实体间关系的敏感性更强

###### 获取邻域表征时采用注意力机制

在获取邻域表征时，我们引入注意力机制:$e_{N_h}=Σ(h,r,t)_{∈N_h}π(h,r,t)e_t$

其中$\pi(h,r,t)$为注意力因子，控制三元组每次传播的衰减因子，说明有多少信息量从t经过r传向h.

注意力得分如下:$π(h,r,t)=(W_re_t)^Ttanh((W_re_h+e_r))$,注意力得分依赖于关系空间r上的$e_h$和$e_t$的距离（为更接近的实体传播更多的信息）与$GCN$、$GraphSage$将两个节点之间的衰减因子定义为 $\frac{1}{\sqrt{|N_h||N_t|}}$或$\frac{1}{|N_t|}$不同，该注意力机制不仅利用图的邻近结构，还规定了相邻节点的不同重要性，并且还将节点之间的关系进行建模，在传播过程中编码更多的信息

对注意力得分进行$softmax$归一化，得到:$\pi(h,r,t) = \frac{exp(\pi(h,r,t))}{\sum_{(h,r',t')∈N_h}exp(\pi(h,r',t'))}$

在GNN_based中添加注意力更新函数

```python
    def update_attention_batch(self, h_list, t_list, r_idx):#按批次更新注意力层
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list


    def update_attention(self, h_list, t_list, r_list, relations):#A_in(注意力因子)
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)
 
    def forward(self, *input, mode):
		......
        if mode == 'update_att':
            return self.update_attention(*input)
```

在KG和CF训练完成后，对注意力层进行更新

```python
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')
```



