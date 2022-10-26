import requests
import re
import fake_useragent
from multiprocessing.dummy import Pool
from lxml import etree

MOVIE_URL_PREFIX = "https://movie.douban.com/subject/"


def xpath_parser(response):
    content = response.content.decode('utf-8')
    tree = etree.HTML(content)
    movie_info = {}

    # 主要信息：片名 导演 编剧 演员
    title = tree.xpath("/html/body/div[3]/div[1]/h1/span[1]")[0].text.split(" ", 1)
    director = [_.text for _ in
                tree.xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[1]/div[1]/div[1]/div[2]/span[2]/span[2]/*")]
    author = [_.text for _ in
              tree.xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[1]/div[1]/div[1]/div[2]/span[1]/span[2]/a")]
    actor = [_.text for _ in
             tree.xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[1]/div[1]/div[1]/div[2]/span[3]/span[2]/*")]

    # 其他信息：电影类型 制作地区 语言 上映时间等(由于这些信息难以/无法用xpath直接定位，故选择正则表达式来解析)
    other_info = ''.join([_ for _ in tree.xpath("//*[@id=\"info\"]/descendant-or-self::text()")])
    official_web = re.findall("(?<=官方网站: ).[^\n]*", other_info)
    movie_type = re.findall("(?<=类型: ).[^\n]*", other_info)[0].split(" / ")  # 不同类型之间用“ / ”分隔,需要将其切割开
    make_era = re.findall("(?<=制片国家/地区: ).[^\n]*", other_info)[0].split(" / ")
    lang = re.findall("(?<=语言: ).[^\n]*", other_info)[0]
    release_time = re.findall("(?<=片长: ).[^\n]*", other_info)[0].split(" / ")
    alias = re.findall("(?<=又名: ).[^\n]*", other_info)[0].split(" / ")
    IMDb_id = re.findall("(?<=IMDb: ).[^ \n]*", other_info)[0]

    # 评分信息：评分人数 评分（xx星/xx分） 评星分布(评分为1~5星的各有%xx) 评价分布（好于%xx的XX类电影）
    review_num = \
        tree.xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div[1]/div[2]/div/div[2]/a/span")[
            0].text
    rank_mark = tree.xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div[1]/div[2]/strong")[0].text
    # 评分星级数藏在class属性中，例如星级为3.5星, 该位置元素的class就为"bigstar35" 35表示3.5星
    rank_star = str(
        tree.xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div[1]/div[2]/div/div[1]/@class")[
            0]).split("bigstar")[2]
    review_distribute = [_.text for _ in tree.xpath(
        "/html/body/div[3]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div[1]/div[3]/*/span[2]")]
    evaluation = [_.text for _ in tree.xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div[2]/a")]

    # 简介
    description = tree.xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[3]/div/span[1]")[0].text.strip("\n").strip(" ")

    # 职员表
    staff_name = tree.xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[5]/ul/*/div/span[1]/a")
    job_position = tree.xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[5]/ul/*/div/span[2]")
    staff_info = [{"name": staff_name[i].text,
                   "job_position": job_position[i].text} for i in range(0, len(staff_name))]

    movie_info.append({
        "title": title,
        "director": director,
        "author": author,
        "actor": actor,
        "official_website": official_web,
        "movie_type": movie_type,
        "film_set": make_era,
        "lang": lang,
        "release_time": release_time,
        "alias": alias,
        "IMDb_id": IMDb_id,
        "review_num": review_num,
        "review_distribute": review_distribute,
        "eval": evaluation,
        "description": description,
        "staff_info": staff_info
    })
    return movie_info

def spider(URL):
    headers = {'User-Agent': str(fake_useragent.UserAgent().random)}
    response = requests.get(URL, headers=headers)
    ID = re.findall("(?<=/)[0-9]*", URL)[0]
    movie_info = xpath_parser(response).append({"id": ID})

def get_movie_url_pool(file_path):
    with open(file_path, 'r') as f:
        movie_url_list = [MOVIE_URL_PREFIX + movie_id.rstrip('\n') + "/" for movie_id in f]
        return movie_url_list

if __name__ == "__main__":
    movie_url_pool = get_movie_url_pool("./Movie_id.txt")
    spider_pool = Pool(5)
    spider_pool.map(spider, movie_url_pool)
