import json

with open('../stage1/Book_info.json', 'r') as book_file:
    books = json.loads(book_file.read())
    index = 1
    for book in books:
        filename = 'book{:0>3d}'.format(index)
        with open('./books/' + filename + '.json', 'w') as fp:
            json.dump(book, fp)
        index += 1

with open('../stage1/Movie_info.json', 'r') as movie_file:
    movies = json.loads(movie_file.read())
    index = 1
    for movie in movies:
        filename = 'movie{:0>3d}'.format(index)
        with open('./movies/' + filename + '.json', 'w') as fp:
            json.dump(movie, fp)
        index += 1
        