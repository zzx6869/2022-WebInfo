import numpy as np
import math

def binary(x,l=1):
    fmt = '{0:0%db}' % l
    return fmt.format(x)

def unary(x):
    return x*'1'+'0'

def elias_generic(lencoding, x):
    if x == 0: return '0'
    l = int(np.log2(x))
    a = x - 2**(int(np.log2(x)))
    k = int(np.log2(x))
    return lencoding(l) + binary(a,k)

def golomb(b, x):
    q = int((x) / b)
    r = int((x) % b)
    l = int(math.ceil(np.log2(b)))
    return unary(q) + binary(r, l)

def elias_gamma(x):
    return elias_generic(unary, x)

def elias_delta(x):
    return elias_generic(elias_gamma,x)

def add_leading_zeros(gamma_code):
    while len(gamma_code) % 8 != 0:
        gamma_code = '0' + gamma_code
    return gamma_code

def split_code_to_8_bits(code):
    bytes_of_code = []
    for i in range(len(code) // 8):
        bytes_of_code.append(code[i*8:(i+1)*8])
    return bytes_of_code