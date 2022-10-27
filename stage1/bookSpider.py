import requests
import re
import fake_useragent
import random
from multiprocessing.dummy import Pool
from lxml import etree
import pymongo
from time import sleep
import json
BOOK_URL_PREFIX = "https://book.douban.com/subject/"


def page_parser(response):
    content = response.content.decode('utf-8')
    content = content.replace("\n", "")
    tree = etree.HTML(content)
    book_info = {}
    info = ''.join(tree.xpath("//*[@id=\"info\"]/descendant-or-self::text()")).replace("\n", " ")
    ID = re.findall("(?<=/)[0-9]+", response.url)
    author = re.findall("(?<=name\": \").+?(?=\")", content)
    name = re.findall("(?<=name\" : \").+?(?=\")", content)
    publishing_house = re.findall("(?<=出版社:)[ ]*[^ ]+", info)
    produce_year = re.findall("(?<=出版年:)[ ]*[^ ]+", info)
    page_num = re.findall("(?<=页数:)[ ]*[^ ]+", info)
    price = re.findall("(?<=定价:)[ ]*[^ ]+", info)
    binding = re.findall("(?<=装帧:)[ ]*[^ ]+", info)
    series = re.findall("(?<=丛书:)[ ]*[^ ]+", info)
    translator = re.findall("(?<=译者:)[ ]*[^ ]+", info)
    ISBN = re.findall("(?<=ISBN:)[ ]*[^ ]+", info)
    rating_info = tree.xpath("//div[@class=\"rating_wrap clearbox\"]/*")
    rating_sum = ""\
        .join(tree.xpath("//div[@class=\"rating_sum\"]/descendant-or-self::text()"))\
        .replace("人评价", "")\
        .strip(" ")
    rating_mark = tree.xpath("//strong[@class=\"ll rating_num \"]")[0].text
    rating_distribute = [_.text for _ in tree.xpath("//div[@class=\"rating_wrap clearbox\"]/span[@class='rating_per']")]

    description_of_book = "".join(tree.xpath("(//div[@class=\"intro\"])[2]/descendant-or-self::text()"))

    if re.findall("(?<=作者简介).+", content):
        description_of_author = "".join(tree.xpath("(//div[@class=\"intro\"])[last()]/descendant-or-self::text()"))
    else:
        description_of_author = []

    book_info = {
        "id": ID,
        "name": name,
        "author": author,
        "publish_house": publishing_house,
        "produce_year": produce_year,
        "page_num": page_num,
        "price": price,
        "binding": binding,
        "series": series,
        "ISBN": ISBN,
        "review_num": rating_sum,
        "review_distribute": rating_distribute,
        "rank_mark": rating_mark,
        "description_of_book": description_of_book,
        "description_of_author": description_of_author,
    }
    return book_info

def spider(URL, cookie, dataBase=None, sleep_time=5):
    headers = {'User-Agent': str(fake_useragent.UserAgent().random)}
    response = requests.get(URL, headers=headers, cookies=cookie)
    book_info = page_parser(response)
    if dataBase is not None:
        dataBase.insert_one(book_info)
    sleep(sleep_time)

if __name__ == "__main__":
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    DB = client["DOUBAN"]
    Book_db = DB["Book"]
    with open("./Movie_id.txt", 'r') as f:
        book_url_list = [BOOK_URL_PREFIX + movie_id.rstrip('\n') + "/" for movie_id in f]
    with open('cookie.json', 'r', encoding='utf-8') as a:
        cookie = json.load(a)
    for url in book_url_list:
        spider(url, cookie, dataBase=Book_db, sleep_time=10)
