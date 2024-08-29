# https://eprint.iacr.org/2012/144.pdf

import logging
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial import polynomial as poly

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='[%(levelname)s] %(message)s\n',
    datefmt='%Y-%m-%d %H:%M:%S',  # Format for date and time
)

# Create a logger object
logger = logging.getLogger()

class Scheme():

    # Initialize BFV scheme parameters
    def __init__(self):
        self.n = 16 # Polynomial degree
        self.t = 12 # Plaintext coefficient modulus
        self.q = 8192  # Ciphertext coefficient modulus
        self.p = self.q ** 3 
        self.m = 0 # Error sampling mean
        self.s = 0 # Error sampling variance
        self.b = 6 * self.s # Error sampling bound
        self.croot = Polynomial(np.array([1] + [0] * (self.n-1) + [1])) # Cyclotomic root


# GLOBAL params
config = Scheme()


class Plaintext():

    def __init__(self, m):
        """
        Initialize a plaintext polynomial.

        Parameters:
            m (int): The integer message to encode.
        """
        self.m = m # Integer message
        self.n = config.n # Polynomial degree
        self.bin = format(m, 'b') # Binary message representation
        self.poly = self.encode() # Plaintext space message representation


    def encode(self):
        """
        Encode a message into plaintext space.
        """
        bin_pad = self.bin.rjust(config.n, '0')[::-1] # Append zeros for up to degree N
        coeffs = list(int(bit) for bit in bin_pad) # Create polynomial coefficients
        return Polynomial(np.array(coeffs)) # Return as numpy array


class Ciphertext():

    def __init__(self, m: Plaintext=None, b=None, a=None, init=True):
        """
        Initialize a ciphertext polynomial.

        Parameters:
            m (int): The plaintext polynomial to encrypt.
            b (np.array): Public key (b)
            a (np.array): Public key (a)
        """
        if init:
            self.m = m # Plaintext polynomial
            self.poly = self.encrypt(b, a) # Ciphertext polynomial

    def encrypt(self, b, a):
        """
        Encode a plaintext polynomial into ciphertext space.

        Parameters:
            b (np.array): Public key (b)
            a (np.array): Public key (a)
        """
        u = binary_poly(config.n)
        e1 = normal_poly(config.n, config.m, config.s)
        e2 = normal_poly(config.n, config.m, config.s)
        dm = self.m.poly * np.floor(config.q / config.t)
        c1 = b * u + e1 + dm
        c1 = modq(c1)
        c2 = a * u + e2
        c2 = modq(c2)
        self.c1, self.c2 = c1, c2
        return (c1, c2)

    def decrypt(self, sk):
        """
        Decrypt a ciphertext polynomial into plaintext space.

        Parameters:
            sk (np.array): Secret key
        """
        m = modq(self.c1 + self.c2 * sk)
        m = (config.t * m) / config.q
        m = Polynomial(np.round(m.coef))
        m = modt(m)
        print(m)
        return m


def modt(x):
    return Polynomial(poly.polydiv(x.coef, config.croot.coef)[1] % config.t)
def modq(x):
    return Polynomial(poly.polydiv(x.coef, config.croot.coef)[1] % config.q)
def modpq(x):
    return Polynomial(poly.polydiv(x.coef, config.croot.coef)[1] % (config.q *
                      config.p))

def binary_poly(size):
    """
    Generate a ternary polynomial with coefficients in {-1,0,1}
    
    Parameters:
        size (int): Desired polynomial degree
    """
    return Polynomial(np.random.randint(0, 2, size, dtype=np.int64))


def uniform_poly(size, modulus):
    """
    Generate a uniformly sampled polynomial in R_<modulus>
    
    Parameters:
        size (int): Desired polynomial degree
        modulus (int): Maximum coefficient
    """
    return Polynomial(np.random.randint(0, modulus, size, dtype=np.int64))


def normal_poly(size, mean, sigma):
    """
    Generate a discrete gaussian sampled polynomial
    
    Parameters:
        size (int): Desired polynomial degree
        mean (int): Mean
        sigma (float): Variance
    """
    return Polynomial(np.clip(
                np.int64(
                    np.random.normal(
                        mean,
                        sigma,
                        size=size
                    )
                ), min=np.ceil(config.b/2), max=np.floor(config.b/2)
            ))

def add(x, y):

    x_1, x_2 = x.poly
    y_1, y_2 = y.poly
    sum_1 = modq(x_1 + y_1)
    sum_2 = modq(x_2 + y_2)
    res = Ciphertext(init=False)
    res.c1, res.c2 = sum_1, sum_2
    return res

def mul(x, y):

    x1, x2 = x.poly
    y1, y2 = y.poly

    # Compute the tensor product and scale
    c1 = (config.t * (x1 * y1)) / config.q
    c2 = (config.t * (x1 * y2 + x2 * y1)) / config.q
    c3 = (config.t * (x2 * y2)) / config.q

    c1 = Polynomial(np.round(c1.coef))
    c2 = Polynomial(np.round(c2.coef))
    c3 = Polynomial(np.round(c3.coef))

    c1 = modq(c1)
    c2 = modq(c2)
    c3 = modq(c3)

    return (c1, c2, c3)

def relin(c1, c2, c3, evk):

    c3_0 = modq(Polynomial(np.round(((c3 * evk[0]) / config.p).coef)))
    c3_1 = modq(Polynomial(np.round(((c3 * evk[1]) / config.p).coef)))

    c1_hat = modq(c1 + c3_0)
    c2_hat = modq(c2 + c3_1)

    res = Ciphertext(init=False)
    res.c1, res.c2 = c1_hat, c2_hat
    return res

def key_gen():
    """
    Generate a secret key and public keys.
    """
    # Secret
    sk = binary_poly(config.n)

    # Public
    a = uniform_poly(config.n, config.q)
    e = normal_poly(config.n, config.m, config.s)
    b = -((a * sk) + e)
    b = modq(b)


    # Eval
    a_evk = uniform_poly(config.n, config.q * config.p)
    e = normal_poly(config.n, config.m, config.s)
    evk = -1 * (a_evk * sk + e) + config.p * (sk * sk)
    evk = modpq(evk)

    return (b, a), sk, (evk, a_evk)


if __name__ == '__main__':

    d = 5

    # Generate keys
    pk, sk, eval = key_gen()
    logger.debug(f"Secret Key:\n{sk}\n")
    logger.debug(f"Public Key 1:\n{pk[0]}\n")
    logger.debug(f"Public Key 2:\n{pk[1]}\n")

    # Encode message
    plain = Plaintext(d)
    logger.info(f"Plaintext Message:\n{d}\n")
    logger.debug(f"Encoded Message:\n{plain.poly}\n")

    # Encrypt message
    cipher = Ciphertext(plain, pk[0], pk[1])
    logger.debug(f"Ciphertext 1:\n{cipher.poly[0]}\n")
    logger.debug(f"Ciphertext 2:\n{cipher.poly[1]}\n")

    # Decrypt message
    plain = cipher.decrypt(sk)
    logger.debug(f"Decryption:\n{plain}\n")

    # Decode message
    logger.info(f"Decoding:\n{int(plain(2))}\n")
    
    sum = add(cipher, cipher).decrypt(sk)
    
    # Decode sum
    logger.info(f"Decoding:\n{int(sum(2))}\n")

    prod = mul(cipher, cipher)
    c1, c2, c3 = prod
    prod = relin(c1, c2, c3, eval).decrypt(sk)

    logger.info(f"Decoding:\n{int(prod(2))}\n")
