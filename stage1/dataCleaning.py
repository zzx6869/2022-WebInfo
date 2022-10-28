import json
import re
import fastJson




if __name__ == "__main__":
    with open("Movie_info.json", "r") as f:
        raw = json.load(f)
    for i in range(0, len(raw)):
        for key, value in raw[i].items():
            if key != 'review_distribute' and key != 'movie_name' and key != 'better_than' and key != 'staff_info':
                raw[i][key] = "".join(value)
            if key == 'director' or key == 'author' or key == 'actor' or key == 'movie_type' or key == 'lang' or key == 'alias' or key == 'film_set' or key == "release_time" or key == 'duration':
                raw[i][key] = raw[i][key].split("/")
            if key == 'description':
                raw[i][key] = raw[i][key].replace("\n", "").replace(" ", "")
    with open("movie_data.json", "w") as f:
        json.dump(raw, f)

