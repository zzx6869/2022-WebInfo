import json

with open('../stage1/Book_info.json', 'r') as book_file:
    books = json.loads(book_file.read())
    index = 0
    for book in books:
        filename = 'book{:0>3d}'.format(index)
        with open('./books/' + filename + '.json', 'w') as fp:
            json.dump(book, fp)
        index += 1
        