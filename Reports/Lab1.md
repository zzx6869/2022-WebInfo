# WebInfo Lab1

> 小组成员：
>
> 张展翔 PB20111669
>
> 黄鑫 PB20061174
>
> 刘阳 PB20111677

## Stage 1 爬虫爬取豆瓣电影，书籍信息

### 实验目标

​	针对给定的电影、书籍ID，爬取其豆瓣主页，并解析其基本信息，获取电影数据的其基本信息、剧情简介、演职员表，以及书籍数据的基本信息、内容简介、作者简介.

### 实验流程分析

#### 爬取工具

在python3.9环境下使用requests包对指定页面进行请求和分析.

主要环境/工具:

- python 3.9
- requests(请求页面)
- fake_useragent(生成伪造UA)
- lxml(xpath解析页面信息)
- re(正则解析页面信息)
- threading(多线程实现)

#### 反爬措施与应对策略

实验中所遇到的网站反爬措施包括：

##### UA检测：对有异常UA的请求，无法正常访问页面

解决方案：使用fake_useragent，每次请求后更换一次UA.

```python
header['User-Agent'] = str(fake_useragent.UserAgent().random)
```

##### IP检测：滑动验证码验证/扫码验证登陆: 同一IP下当爬取速度达到一定阈值，就需要通过扫码/滑动验证码验证并登陆后才能正常访问.

解决方案：

- 控制爬取速度，在成功爬取一个URL后使爬虫sleep指定时间间隔防止触发反爬机制(失败，出于某种原因还是会被检测到)
- 手动登陆获取Cookie(失败,Cookie有时限，且登录状态下访问到达一定速度/数量也会被检测并触发滑动验证码验证)
- 使用selenium,在触发滑动验证码验证时模拟鼠标滑动破解(可靠性和性能低，不采用)
- 使用高匿IP代理池, 每爬取一定数量/被检测到时就更换一次代理IP, 突破检测，同时也解决了爬虫的速度限制，可以使用多线程或异步爬虫的方式来进一步提高速度(成功，效果极佳)

通过API获取并返回代理IP(代码中接口已失效),通过该代理进行请求

```python
def get_proxy():
    try:
        PROXY_API_URL = "https://h.shanchendaili.com/api.html?action=get_ip&key=HUe820e7971028093030zNPr&time=10" \
                        "&count=1&protocol=http&type=json&only=1 "
        proxy_json = requests.get(PROXY_API_URL).text
        server = re.findall("(?<=\"sever\":\").*?(?=\")",
                            proxy_json)[0]
        port = re.findall("(?<=\"port\":).*?(?=,)",
                          proxy_json)[0]
        host = server + ":" + port
    except:
        return None
    proxy = {
        'http': 'http://' + host,
        'https': 'https://' + host,
    }
    return proxy
```

爬取失败更换代理ip

```python
try:
                response = requests.get(_url, headers=header, cookies=cookie, proxies=proxy, timeout=8)
                movie_info = page_parser_movie(response)
                break
except Exception as e:  # 同一ip爬取失败达到3次则更换一次ip,若再失败就放弃该url
                print("\n")
                print(e)

                if same_proxy_retry < 3:
                    print("Crawling error:retrying for {re} time(s)\n".format(re=same_proxy_retry + 1))
                    same_proxy_retry += 1
                    continue

                else:
                    same_proxy_retry = 0
                    retry += 1
                    print("Crawling error:change proxy\n".format(re=retry + 1))
                    proxy = get_proxy()
```

#### 页面解析方法

​	由于直接使用正则表达式，表达式编写较为繁琐且程序鲁棒性低.而各个电影之间基本信息的页面排布属性不同，故直接通过xpath获取也较为低效不合适.故爬虫采用***xpath解析和正则表达式解析结合***的方式获取页面信息.

​	以电影的基本信息为例,由于基本信息主要排布在id为info的区域中,使用lxml中带有的xpath解析器获取该区域本身及所有子结点的信息并转化为string，再交由正则表达式解析

<img src=".\pic\info.png" alt="info" style="zoom:50%;" />

```python
    other_info = ''.join([_ for _ in tree.xpath("//*[@id=\"info\"]/descendant-or-self::text()")])
    # 获取属性id="info"的结点信息，将其本身与其所有子节点的信息转化为string
    ...
        "director": re.findall("(?<=导演: ).[^\n]*", other_info),
        "author": re.findall("(?<=编剧: ).[^\n]*", other_info),
        "actor": re.findall("(?<=主演: ).[^\n]*", other_info),
        "official_website": re.findall("(?<=官方网站: ).[^\n]*", other_info),
        "movie_type": re.findall("(?<=类型: ).[^\n]*", other_info),
        "film_set": re.findall("(?<=制片国家/地区: ).[^\n]*", other_info),
        "lang": re.findall("(?<=语言: ).[^\n]*", other_info),
        "release_time": re.findall("(?<=上映日期: ).[^\n]*", other_info),
        "duration": re.findall("(?<=片长: ).[^\n]*", other_info),
        "alias": re.findall("(?<=又名: ).[^\n]*", other_info),
        "IMDb_id": re.findall("(?<=IMDb: ).[^ \n]*", other_info),
        "rank_mark": re.findall("(?<=ratingValue\": \").*?(?=\")", content),
        "review_num": re.findall("(?<=ratingCount\": \").*?(?=\")", content),
```

## 结果展示

​	爬虫共爬取到988份指定id的书籍信息以及995项指定id的电影信息，共有12份无效数据ID和5份无效电影ID

​	以下为无效的电影/书籍URL

```python
# 无效的电影URL
'https://movie.douban.com/subject/1310174/', broken
'https://movie.douban.com/subject/1309046/', broken
'https://movie.douban.com/subject/1768351/', broken
'https://movie.douban.com/subject/1796939/', broken
'https://movie.douban.com/subject/1295428/', broken
```

```python
# 无效的书籍URL
'https://book.douban.com/subject/1394253/', broken
'https://book.douban.com/subject/4112874/', broken
'https://book.douban.com/subject/1059336/', broken
'https://book.douban.com/subject/1456034/', broken
'https://book.douban.com/subject/1918734/', broken
'https://book.douban.com/subject/1025723/', broken
'https://book.douban.com/subject/1903968/', broken
'https://book.douban.com/subject/1079509/', broken
'https://book.douban.com/subject/4886245/', broken
'https://book.douban.com/subject/1767388/', broken
'https://book.douban.com/subject/1051363/', broken
'https://book.douban.com/subject/1803022/', broken	
```

​	爬取到的每个项目的信息以dict list的形式保存在json文件中，每个列表项包含了书籍/电影的ip以及指定爬取信息的字典.以下为爬取到的信息展示（此为未经过进一步清洗的数据，数据清洗在后续实现，此处仅作展示.使用MongoDB Compass展示）

#### 电影信息展示

<img src=".\pic\movie.png" alt="movie" style="zoom: 67%;" />

#### 书籍信息展示

<img src=".\pic\book.png" alt="book" style="zoom: 80%;" />

## Stage 2

## Stage 3

### 实验目标

基于第一阶段爬取的豆瓣 Movie/Book 信息、我们提供的豆瓣电影与书籍的评分记 录以及用户间的社交关系，判断用户的偏好

### 实验内容

本次评分的预测只利用了Movie_score.csv文件评分部分，没有使用contract文件

#### 测试集与训练集划分

训练集和测试集划分比例为8：2

利用sklearn库中的model_selection_train_test_split函数完成划分

#### 预测相关计算方法

预测方法采用基于用户的均值中心化算法来进行计算
$$
pred(u,i)=\hat{r_{ui}}=\bar{r_u}+\frac{\Sigma_{v\in U}sim(u,v)*(r_{vi}-\bar{r_v})}{\Sigma_{v\in U}{|sim(u,v)|}}
$$
其中，用户间的相似度$sim(u,v)$则利用了皮尔逊相关系数来计算
$$
\rho_{X,Y}=\frac{cov(X,Y)}{\sigma_X\sigma_Y}=\frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X\sigma_Y}
$$
对于仅有少数正常评分或者全部评分为0的用户，无法计算其与其他用户的相似度，故舍去不做预测

#### ndcg计算

最后，使用sklearn库中的metrics.ndcg_score函数完成对ndcg的计算

### 结果展示

#### 文件结构



![tree](./pic/tree.png)

- Movie_score.csv为提供电影评分样本
- **all.csv为最后预测评分与实际评分汇总**
- cache.csv为用户-电影评分矩阵
- data_process.py为数据处理代码
- forall为总的评分样本汇总文件（未使用）
- ndcg.py为ndcg计算代码
- pic为截图保存文件夹
- similar_cache.csv为用户相关度矩阵
- test.csv和train为测试集和训练集
- train.py为预测评分代码

#### nDCG计算

如下图为测试机中实际分数与预测分数的对应表

![image-20221120154001713](.\pic\predict.png)

经计算可得

ndcg=0.9814199692967567

ndcg@5=0.9815151048429968

ndcg@10=0.9421806099621497
