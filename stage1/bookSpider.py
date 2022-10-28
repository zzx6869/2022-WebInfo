import requests
import re
import fake_useragent
from lxml import etree
from queue import Queue
from threading import Lock
from time import sleep
import threading
import json
BOOK_URL_PREFIX = "https://book.douban.com/subject/"

def page_parser_book(response):
    content = response.content.decode('utf-8')
    content = content.replace("\n", "")
    tree = etree.HTML(content)
    info = ''.join(tree.xpath("//*[@id=\"info\"]/descendant-or-self::text()")).replace("\n", " ")
    ID = re.findall("(?<=/)[0-9]+", response.url)
    author = re.findall("(?<=name\": \").+?(?=\")", content)
    name = re.findall("(?<=name\" : \").+?(?=\")", content)
    origin_name = re.findall("(?<=原作名:).*?(?=[\S]+:)", info)
    vice_name = re.findall("(?<=副标题:).*?(?=[\S]+:)", info)
    publishing_house = re.findall("(?<=出版社:)[ ]*[^ ]+", info)
    produce_year = re.findall("(?<=出版年:)[ ]*[^ ]+", info)
    page_num = re.findall("(?<=页数:)[ ]*[^ ]+", info)
    price = re.findall("(?<=定价:)[ ]*[^ ]+", info)
    binding = re.findall("(?<=装帧:)[ ]*[^ ]+", info)
    series = re.findall("(?<=丛书:)[ ]*[^ ]+", info)
    translator = re.findall("(?<=译者:)[ ]*[^ ]+", info)
    ISBN = re.findall("(?<=ISBN:)[ ]*[^ ]+", info)
    rating_sum = ""\
        .join(tree.xpath("//div[@class=\"rating_sum\"]/descendant-or-self::text()"))\
        .replace("人评价", "")\
        .strip(" ")
    rating_mark = tree.xpath("//strong[@class=\"ll rating_num \"]")[0].text
    rating_distribute = [_.text for _ in tree.xpath("//div[@class=\"rating_wrap clearbox\"]/span[@class='rating_per']")]

    if re.findall("(?<=内容简介).+", content):
        description_of_book = "".join(tree.xpath("(//div[@class=\"intro\"])[1]/descendant-or-self::text()"))
    else:
        description_of_book = []

    if re.findall("(?<=作者简介).+", content):
        description_of_author = "".join(tree.xpath("(//div[@class=\"intro\"])[last()]/descendant-or-self::text()"))
    else:
        description_of_author = []

    book_info = {
        "id": ID,
        "name": name,
        "vice_name": vice_name,
        "ori_name": origin_name,
        "author": author,
        "publish_house": publishing_house,
        "produce_year": produce_year,
        "page_num": page_num,
        "price": price,
        "binding": binding,
        "translator": translator,
        "series": series,
        "ISBN": ISBN,
        "review_num": rating_sum,
        "review_distribute": rating_distribute,
        "rank_mark": rating_mark,
        "description_of_book": description_of_book,
        "description_of_author": description_of_author,
    }
    return book_info

def get_proxy():
    while 1:
        try:
            PROXY_API_URL = "https://h.shanchendaili.com/api.html?action=get_ip&key=HUe820e7971028093030zNPr&time=10" \
                            "&count=1&protocol=http&type=json&only=1 "
            proxy_json = requests.get(PROXY_API_URL).text
            server = re.findall("(?<=\"sever\":\").*?(?=\")",
                                proxy_json)[0]
            port = re.findall("(?<=\"port\":).*?(?=,)",
                              proxy_json)[0]
            host = server + ":" + port
            break
        except:
            host = None
            print("Can't fetch proxy. Retrying\n")
            sleep(3)
            continue
    proxy = {
        'http': 'http://' + host,
        'https': 'https://' + host,
    }
    return proxy

def spider(header, lock, queue, finish_num, cookie=None):
    proxy = get_proxy()

    while not queue.empty():

        _url = queue.get()
        same_retry = 0
        retry = 0
        book_info = None

        while retry < 2:
            try:
                response = requests.get(_url, headers=header, cookies=cookie, proxies=proxy, timeout=8)
                book_info = page_parser_book(response)
                break
            except Exception as e:
                print("\n")
                print(e)
                if same_retry < 3:
                    print("Parser error:retrying for the {re} time(s)\n".format(re=same_retry + 1))
                    same_retry += 1
                    continue
                else:
                    print("Parser error:change proxy\n".format(re=retry + 1))
                    proxy = get_proxy()
                    retry += 1
                    same_retry = 0

        if book_info is not None:
            lock.acquire()
            with open("./Book_info.json", 'a') as f:
                json.dump([book_info], f, indent=4)
            with open("./Checkpoint_book.txt", "a") as ckpt:
                ckpt.write("\n" + _url)
            finish_num[0] += 1
            print('\nProcess:{done}/{Total}\n'.format(done=finish_num[0], Total=1000))
            lock.release()

if __name__ == "__main__":
    with open("./Book_id.txt", 'r') as f:
        book_url_list = [BOOK_URL_PREFIX + movie_id.rstrip('\n') + "/" for movie_id in f]

    with open("./Checkpoint_book.txt", 'r+', encoding='utf-8') as f:
        url_complete = [_.strip("\n") for _ in f]

    # try:
    #     with open('cookie.json', 'r', encoding='utf-8') as a:
    #         cookie = json.load(a)
    # except:
    #     cookie = None

    book_url_list = list(set(book_url_list) - set(url_complete))
    Book_queue = Queue()
    for i in book_url_list:
        Book_queue.put(i)

    header = {'User-Agent': str(fake_useragent.UserAgent().random),
               'Host': "book.douban.com",
               "Sec-Fetch-Dest": "document",
               "Sec-Fetch-Mode": "navigate",
               "Sec-Fetch-Site": "none",
               "Sec-Fetch-User": "?1"}

    t_list = []
    finish_num = [0]
    MAX_THREAD_NUM = 8
    lock = Lock()
    for i in range(0, MAX_THREAD_NUM):
        t = threading.Thread(target=spider, args=[header, lock, Book_queue, finish_num])
        t_list.append(t)
        t.start()
    for t in t_list:
        t.join()

