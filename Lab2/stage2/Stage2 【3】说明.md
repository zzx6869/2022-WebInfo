# Stage2 【3】说明

## KG的构建

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

## TransE,TransR算法的实现

### TransE

TransE即将关系视为了向量空间中头实体和尾实体之间的操作

首先，需要对关系嵌入、头实体嵌入、尾实体嵌入以及负采样的尾实体嵌入进行L2归一化处理

可以采用`torch`中的`normalize`函数实现

三元组的得分涉及向量距离的运算，具体方式如下

```python
pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)                                 
neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)       
```

然后使用BPR进行优化，使负样本得分大于正样本

```python
kg_loss =torch.mean ((float)(-1) * F.logsigmoid(neg_score - pos_score))
```

### TransR

TransR是在TransE的投影基础上将其投放到空间上，即将投影向量转为投影矩阵，因此需将h和t投影到r所在的空间中

可以利用torch中的squeeze函数进行维度运算

```python
r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)            
r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)
```

其余处理和TransE中大致类似

## 多任务优化与迭代优化

多任务优化即计算出kg的损失函数和cf的损失函数，并使它们相加，然后一同进行反向传播更新等操作，即代码模板所给的原有方式

迭代优化则需要分开计算出kg的损失函数和cf的损失函数，分别惊醒反向传播更新和清梯度，二者交替进行

故需要更改`main_Embedding_based.py`文件中的train model部分，

将cf_loss和kg_loss分开计算

具体代码如下

```python
for epoch in range(1, args.n_epoch + 1):
        model.train()
        time0=time()
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

            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
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

            if (iter % args.kg_print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))

```

