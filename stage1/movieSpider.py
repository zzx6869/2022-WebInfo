import requests
import re
import fake_useragent
import json
from time import sleep
from lxml import etree
import pymongo
MOVIE_URL_PREFIX = "https://movie.douban.com/subject/"

def page_parser(response):
    content = response.content.decode('utf-8')
    tree = etree.HTML(content)
    other_info = ''.join([_ for _ in tree.xpath("//*[@id=\"info\"]/descendant-or-self::text()")])
    staff_name = tree.xpath("//span[@class=\"name\"]/a")
    job_position = tree.xpath("//span[@class=\"role\"]")
    movie_info = {
        "id": re.findall("(?<=/)[0-9]+", response.url)[0],
        "movie_name": str(tree.xpath("//meta[@property=\"og:title\"]/@content")[0]).split(" ", 1),
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
        "review_distribute": [_.text for _ in tree.xpath("//span[@class='rating_per']")],
        "better_than": [_.text for _ in tree.xpath("//div[@class=\"rating_betterthan\"]/a")],
        "description": tree.xpath("//span[@property=\"v:summary\"]")[0].text.replace(" ", "").replace("\n", ""),
        "staff_info": [{"name": staff_name[i].text,
                        "job_position": job_position[i].text} for i in range(0, len(staff_name))]

    }
    for key in ("alias", "director", "author", "actor", "movie_type", "release_time"):
        movie_info[key] = movie_info[key][0].split(" / ")
    return movie_info

def spider(URL, cookie, dataBase=None, sleep_time=5):
    headers = {'User-Agent': str(fake_useragent.UserAgent().random)}
    response = requests.get(URL, headers=headers, cookies=cookie)
    movie_info = page_parser(response)
    if dataBase is not None:
        dataBase.insert_one(movie_info)
    sleep(sleep_time)


if __name__ == "__main__":
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    DB = client["DOUBAN"]
    Movie_db = DB["Movie"]
    with open("./Movie_id.txt", 'r') as f:
        movie_url_list = [MOVIE_URL_PREFIX + movie_id.rstrip('\n') + "/" for movie_id in f]
    with open('cookie.json', 'r', encoding='utf-8') as a:
        cookie = json.load(a)
    for url in movie_url_list:
        spider(url, cookie, dataBase=Movie_db, sleep_time=10)

