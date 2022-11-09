from encoding import *

class Compressor:
    def __init__(self, index_list):
        self.index_list = index_list

    def build(self):
        compress_data = bytes('', 'ascii')
        for index in self.index_list:
            elias_code = elias_gamma(index)
            compress_data += bytes([int(x, 2) for x in split_code_to_8_bits(add_leading_zeros(elias_code))])
        return compress_data

    def decode(self, data: bytes):
        num = int.from_bytes(data, 'big', signed=False)
        bit_list = []
        for i in range(data.__len__() * 8):
            bit_list.append(num % 2)
            num = num // 2
        bit_list.reverse()
        
        res = []
        i = 0
        while i < bit_list.__len__():
            len = 0
            exp = 0
            offset = 0
            while bit_list[i] == 0:
                i += 1
            while bit_list[i] == 1:
                len += 1
                i += 1
            exp = 1 << len
            while len > 0:
                len -= 1
                i += 1
                offset = (offset << 1) + bit_list[i]
            i += 1
            res.append(exp + offset)
        return res
                