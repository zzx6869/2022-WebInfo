import json
from indexCompressor import Compressor

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
            if a[i] <= b[j]:
                ans.append(a[i])
                i += 1
            else:
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
                i += 1
                j += 1
                ans.append(a[i])
    return ans

def get_list(table: dict, word: str):
    index = table[word]
    with open('./book_keywords/word' + '{:0>4d}'.format(index)) as word_file:
        return Compressor(None).decode(bytes(word_file.read()))

def boolSearch(a: str, op: str, b: str, table: dict):
    if (op == 'and'):
        return extract(get_list(table, a), get_list(table, a))
    elif (op == 'or'):
        return merge(get_list(table, a), get_list(table, a))
    return None

if __name__ == '__main__':
    # TODO
