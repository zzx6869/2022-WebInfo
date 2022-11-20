import requests
import re
import fake_useragent
import threading
import json
from lxml import etree
from queue import Queue
from threading import Lock



def page_parser_movie(response):
    content = response.text
    content = re.sub("<br[ ]*?/>", "", content)
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
        "description": tree.xpath("//span[@class=\"all hidden\"]"),
        "staff_info": [{"name": staff_name[i].text,
                        "job_position": job_position[i].text} for i in range(0, len(staff_name))]
    }
    if not movie_info["description"]:
        movie_info["description"] = tree.xpath("//span[@property=\"v:summary\"]")[0].text
    else:
        movie_info["description"] = movie_info["description"][0].text

    return movie_info


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


def spider(header, lock, queue, finish_num, cookie=None):
    proxy = get_proxy()

    while not queue.empty():
        _url = queue.get()
        same_proxy_retry = 0
        retry = 0
        movie_info = None
        header['User-Agent'] = str(fake_useragent.UserAgent().random)

        while retry < 2:

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

        if movie_info is not None:
            lock.acquire()
            with open("./Movie_info.json", 'a') as info_f:
                json.dump([movie_info], info_f, indent=4)
            with open("./Checkpoint_movie.txt", "a") as ckpt:
                ckpt.write("\n" + _url)
            finish_num[0] += 1
            print('\nProcess:{done}/{Total}\n'.format(done=finish_num[0], Total=1000))
            lock.release()


if __name__ == "__main__":

    MOVIE_URL_PREFIX = "https://movie.douban.com/subject/"

    with open("./Movie_id.txt", 'r') as f:
        book_url_list = [MOVIE_URL_PREFIX + movie_id.rstrip('\n') + "/" for movie_id in f]

    with open("./Checkpoint_movie.txt", 'r+', encoding='utf-8') as f:  # checkpoint保存已经爬取成功的url
        url_complete = [_.strip("\n") for _ in f]

    book_url_list = list(set(book_url_list) - set(url_complete)) # 去掉已经爬取过的部分

    # try:
    #     with open('cookie.json', 'r', encoding='utf-8') as a:
    #         cookie = json.load(a)
    # except:
    #     cookie = None

    Book_queue = Queue()
    for i in book_url_list:
        Book_queue.put(i)

    header = {'User-Agent': str(fake_useragent.UserAgent().random),
              'Host': "movie.douban.com",
              "Sec-Fetch-Dest": "document",
              "Sec-Fetch-Mode": "navigate",
              "Sec-Fetch-Site": "none",
              "Sec-Fetch-User": "?1"}

    thread_list = []
    finish_num = [0]
    MAX_THREAD_NUM = 8
    file_lock = Lock()
    for i in range(0, MAX_THREAD_NUM):
        t = threading.Thread(target=spider, args=[header, file_lock, Book_queue, finish_num])
        thread_list.append(t)
        t.start()
    for thread in thread_list:
        thread.join()
