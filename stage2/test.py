from encoding import *
from indexCompressor import Compressor
import boolSearch

data = Compressor([0, 13, 24, 35]).build()
print(data)
open('./test', 'wb').write(Compressor([0, 13, 24, 35]).build())

with open('./test', 'rb') as word_file:
    print(Compressor(None).decode(bytes(word_file.readline())))
