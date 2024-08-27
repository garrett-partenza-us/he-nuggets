import numpy as np
from numpy.polynomial import Polynomial


d = 4
n, t, q = 64, 64, 64
m, s = 0, 8/(2*np.pi)
b = 6 * s
root = Polynomial([1] + [0]*(n-1) + [1])

class Plaintext():

    def __init__(self, m):
        self.m = m
        self.n = n
        self.bin = format(m, 'b')
        self.poly = self.encode()

    def encode(self):
        print(self.bin)
        bin_pad = self.bin.rjust(n, '0')[::-1]
        print(bin_pad)
        coeffs = list(int(bit) for bit in bin_pad)
        return Polynomial(coeffs)


class Ciphertext():

    def __init__(self, m: Plaintext, pk1, pk2):
        self.m = m
        self.poly = self.encrypt(pk1, pk2)

    def encrypt(self, pk1, pk2):
        u = gen_r2()
        e1, e2 = gen_error(), gen_error()
        c1 = pk1 * u + e1 + self.m.poly * np.floor(q/t)
        c1 = mod(c1, q)
        c2 = pk2 * u + e1
        c2 = mod(c2, q)
        self.c1, self.c2 = c1, c2
        return (c1, c2)

    def decrypt(self, sk):
        num = t * mod((self.c1 + self.c2 * sk), q)
        res = num / q
        res = np.round(num.coef).astype(int)
        res = Polynomial(res)
        res = mod(res, t)
        return res

def mod(poly, q):
    _, remainder = np.polydiv(poly.coef, root.coef)
    return Polynomial(np.mod(remainder, q))

def gen_r2():
    coeffs = np.random.randint(-1, 2, n)
    return Polynomial(coeffs)

def gen_error():
    coeffs = np.random.normal(m, s, n)
    coeffs = np.round(coeffs).astype(int)
    coeffs = np.clip(coeffs, -b, b)
    return Polynomial(coeffs)


def gen_pk(sk):
    a_coeffs = np.random.randint(np.ceil(-q/2), np.floor(q/2), n+1)
    a = Polynomial(a_coeffs)
    pk1 = (-1 * (a * sk + gen_error()))
    pk1 = mod(pk1, q)
    return (pk1, a)


if __name__ == '__main__':


    sk = gen_r2()
    pk1, pk2 = gen_pk(sk)
    plain = Plaintext(d)
    print(plain.poly)
    print(plain.poly(2))
    cipher = Ciphertext(plain, pk1, pk2)
    plain = cipher.decrypt(sk)
    print(plain(2))
