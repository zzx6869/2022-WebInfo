import thulac
import json
from indexCompressor import Compressor


def get_wordlist():
    lac1 = thulac.thulac(model_path='../../models', seg_only=False, filt=True)
    briefs = []
    stop_words = [line.strip() for line in open('baidu_stopwords.txt', 'r').readlines()]
    stop_words.append(' ')
    with open('../stage1/Book_info.json', 'r') as book_file:
        books = json.loads(book_file.read())
        for book in books:
            message = book['name'][0]
            if (isinstance(book['description_of_book'], str)):
                message += book['description_of_book']
            if (isinstance(book['description_of_author'], str)):
                message += book['description_of_author']
            words = lac1.cut(message)
            for word in words.copy():
                if word[0] in stop_words:
                    words.remove(word)
            briefs.append(words)
    return briefs

def makeTable(wordlists):
    table = dict()
    index = 1
    for wordlist in wordlists:
        for word in wordlist:
            if (table.get(word[0]) != None):
                table[word[0]].add(index)
            else:
                table[word[0]] = set()
                table[word[0]].add(index)
        index += 1
    index = 0
    for key in table.keys():
        index_list = list(table[key])
        index_list.sort()
        filename = 'word{:0>4d}'.format(index)
        open('./book_keywords/' + filename, 'wb').write(Compressor(index_list).build())
        table[key] = index
        index += 1
    return table

if __name__ == '__main__':
    wordlists = get_wordlist()
    table = makeTable(wordlists)
    with open('./table.json', 'w') as table_file:
        json.dump(table, table_file)
