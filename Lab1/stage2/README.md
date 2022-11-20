# web-lab1-stage2

> by 刘阳 

## 文件说明

### 算法文件

- boolSearch.py: 向外提供布尔查询的接口，具体使用请参照 `python3 boolSearch.py -h` 。
- encoding.py: 依据 elias gamma 的压缩编码方式，对一个数字进行解码和编码。
- indexCompressor.py: 压缩编码/解码器，对一个词汇的倒排索引进行编码、解码。
- parseFile.py: 对 stage1 得到的 book 和 movie 文件进行处理、再存储。
- tableMaker.py: 生成倒排表。

### 数据文件/文件夹

- book_keywords/* : 存储 book 类型文档各个词汇的倒排索引。
- books/* : 存储各个 book ，用于返回查询的到的文档。
- movie_keywords/* movies/* : 同上。
- book_table.json movie_table.json : 存储两种文档词项的编号，以寻找倒排索引。

## 文档的处理

​	这里直接使用了来自THU的分词算法