from encoding import *
from indexCompressor import Compressor
import boolSearch
import getopt, sys

def main(argv):
    search_type = ''
    search_string = ''
    try:
        opts, args = getopt.getopt(argv, 'ht:s:')
    except getopt.GetoptError:
        print('boolSearch -t search_type -s search_string')
    for opt, arg in opts:
        if (opt == '-h'):
            print('''boolSearch -t search_type -s search_string
        search_type: movie or book     
        search_string: bool expression''')
        elif (opt == '-t'):
            search_type = arg
        elif (opt == '-s'):
            search_string = arg
    print(search_string.split(' ')[-1])

if __name__ == '__main__':
    # TODO
    main(sys.argv[1:])
