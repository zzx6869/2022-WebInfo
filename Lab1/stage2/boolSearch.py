import json
from indexCompressor import Compressor
import sys, getopt


def merge(a: list, b: list):
    [i, j] = [0, 0]
    ans = []
    while i < a.__len__() or j < b.__len__():
        if i == a.__len__():
            ans.append(b[j])
            j += 1
        elif j == b.__len__():
            ans.append(a[i])
            i += 1
        else:
            if a[i] == b[j]:
                ans.append(a[i])
                i += 1
                j += 1
            elif a[i] < b[j]:
                ans.append(a[i])
                i += 1
            elif a[i] > b[j]:
                ans.append(b[j])
                j += 1
    return ans

def extract(a: list, b: list):
    [i, j] = [0, 0]
    ans = []
    while i < a.__len__() or j < b.__len__():
        if i == a.__len__() or j == b.__len__():
            break
        else:
            if a[i] < b[j]:
                i += 1
            elif a[i] > b[j]:
                j += 1
            else:
                ans.append(a[i])
                i += 1
                j += 1
    return ans

def get_list(type: str, table: dict, word: str):
    if word not in table.keys():
        return []
    else:
        index = table[word]
    with open('./' + type + '_keywords/word' + '{:0>5d}'.format(index), 'rb') as word_file:
        return Compressor(None).decode(bytes(word_file.read()))


def boolSearch(a: str, op: str, b: str, table: dict):
    if (op == 'and'):
        return extract(get_list(table, a), get_list(table, a))
    elif (op == 'or'):
        return merge(get_list(table, a), get_list(table, a))
    return None

def main(argv):
    search_type = ''
    search_string = ''
    result_file = ''
    try:
        opts, args = getopt.getopt(argv, 'ht:s:f')
    except getopt.GetoptError:
        print('boolSearch -t search_type -s search_string -f result file')
    for opt, arg in opts:
        if (opt == '-h'):
            print('''boolSearch -t search_type -s search_string -f search file
                        search_type: movie or book     
                        search_string: bool expression
                        result file: default result.txt''')
            exit(0)
        elif (opt == '-t'):
            search_type = arg
        elif (opt == '-s'):
            search_string = arg
        elif (opt == '-f'):
            result_file = arg
    
    table = {}
    if (search_type == 'book'):
        table = json.loads(open('./book_table.json', 'r').read())
    elif (search_type == 'movie'):
        table = json.loads(open('./movie_table.json', 'r').read())
    else:
        print('ERROR: wrong search type.')
        exit(-1)
    
    words = []
    operator = []
    for word in search_string.split():
        if word == 'and' or word == 'or' or word == '(':
            operator.append(word)
        elif word == ')':
            while operator[-1] != '(':
                words.append(operator.pop())
            operator.pop()
        else:
            words.append(word)
    while operator.__len__() != 0:
        words.append(operator.pop())
    print(words)

    cur_list = []
    begin = 0
    tmp = []
    for word in words:
        if word == 'and':
            if begin == 0:
                cur_list = extract(get_list(search_type, table, tmp.pop()), get_list(search_type, table, tmp.pop()))
                begin = 1
            else:
                cur_list = extract(cur_list, get_list(search_type, table, tmp.pop()))
        elif word == 'or':
            if begin == 0:
                cur_list = merge(get_list(search_type, table, tmp.pop()), get_list(search_type, table, tmp.pop()))
                begin = 1
            else:
                cur_list = merge(cur_list, get_list(search_type, table, tmp.pop()))
        else:
            tmp.append(word)
        print(word, cur_list)
    
    if result_file == '':
        result_file = 'result.txt'
    
    with open(result_file, 'w') as file:
        if search_type == 'book':
            for index in cur_list:
                filename = 'book{:0>3d}'.format(index)
                bookInfo = json.loads(open('./books/' + filename + '.json', 'r').read())
                file.write('name: ' + bookInfo['name'][0] + '\n' + 'description:' + bookInfo['description_of_book'] + '\n\n\n')
        elif search_type == 'movie':
            for index in cur_list:
                filename = 'movie{:0>3d}'.format(index)
                bookInfo = json.loads(open('./movies/' + filename + '.json', 'r').read())
                file.write('name: ' + bookInfo['movie_name'][0] + '\n' + 'description:' + bookInfo['description'] + '\n\n\n')


if __name__ == '__main__':
    # TODO
    main(sys.argv[1:])
