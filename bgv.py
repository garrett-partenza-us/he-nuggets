import logging
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial import polynomial as poly

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='[%(levelname)s] %(message)s\n',
    datefmt='%Y-%m-%d %H:%M:%S',  # Format for date and time
)

# Create a logger object
logger = logging.getLogger()

class Scheme():

    # Initialize BFV scheme parameters
    def __init__(self):
        self.n = 64 # Polynomial degree
        self.t = 4 # Plaintext coefficient modulus
        self.q = 65536 # Ciphertext coefficient modulus
        self.m = 0 # Error sampling mean
        self.s = 3.2 # Error sampling variance
        self.b = 6 * self.s # Error sampling bound
        self.croot = np.array([1] + [0] * (self.n-1) + [1]) # Cyclotomic root


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
        return np.array(coeffs) # Return as numpy array


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
        c1 = polymul(b, u, config.q, config.croot)
        c1 = polyadd(c1, e1, config.q, config.croot)
        delta = self.m.poly * np.floor(config.q / config.t)
        delta = poly.polydiv(delta % config.q, config.croot)[1] % config.q
        c1 = polyadd(c1, delta, config.q, config.croot)
        c2 = polymul(a, u, config.q, config.croot)
        c2 = polyadd(c2, e2, config.q, config.croot)
        self.c1, self.c2 = c1, c2
        return (c1, c2)

    def decrypt(self, sk):
        """
        Decrypt a ciphertext polynomial into plaintext space.

        Parameters:
            sk (np.array): Secret key
        """
        m = polymul(self.c2, sk, config.q, config.croot)
        m = polyadd(m, self.c1, config.q, config.croot)
        m = config.t * m
        m = m / config.q
        m = np.round(m).astype(int)
        m = poly.polydiv(m % config.t, config.croot)[1] % config.t
        return m


def polymul(x, y, modulus, poly_mod):
    """
    Multiply two polynomials.
    
    Parameters:
        x (np.array): LHS
        x (np.array): RHS
        modulus (int): Coefficient modulus.
        poly_mod (np.array): Polynomial modulus. 
    """
    return np.int64(
        np.round(
            poly.polydiv(
                poly.polymul(x, y) % modulus,
                poly_mod
            )[1] % modulus
        )
    )


def polyadd(x, y, modulus, poly_mod):
    """
    Add two polynomials.
    
    Parameters:
        x (np.array): LHS
        x (np.array): RHS
        modulus (int): Coefficient modulus.
        poly_mod (np.array): Polynomial modulus. 
    """
    return np.int64(
        np.round(
            poly.polydiv(
                poly.polyadd(x, y) % modulus,
                poly_mod
            )[1] % modulus
        )
    )


def binary_poly(size):
    """
    Generate a ternary polynomial with coefficients in {-1,0,1}
    
    Parameters:
        size (int): Desired polynomial degree
    """
    return np.random.randint(0, 2, size, dtype=np.int64)


def uniform_poly(size, modulus):
    """
    Generate a uniformly sampled polynomial in R_<modulus>
    
    Parameters:
        size (int): Desired polynomial degree
        modulus (int): Maximum coefficient
    """
    return np.random.randint(0, modulus, size, dtype=np.int64)


def normal_poly(size, mean, sigma):
    """
    Generate a discrete gaussian sampled polynomial
    
    Parameters:
        size (int): Desired polynomial degree
        mean (int): Mean
        sigma (float): Variance
    """
    return np.clip(
                np.int64(
                    np.random.normal(
                        mean,
                        sigma,
                        size=size
                    )
                ), min=np.ceil(config.b/2), max=np.floor(config.b/2)
            )

def add(x, y):

    x_1, x_2 = x.poly
    y_1, y_2 = y.poly
    sum_1 = polyadd(x_1, y_1, config.q, config.croot)
    sum_2 = polyadd(x_2, y_2, config.q, config.croot)
    res = Ciphertext(init=False)
    res.c1, res.c2 = sum_1, sum_2
    return res

def mul(x, y):
    
    x1, x2 = x.poly
    y1, y2 = y.poly

    c1n = (polymul(x1, x2, config.q, config.croot) * config.t) / config.q
    c2n = (polyadd(
            polymul(x1, y2, config.q, config.croot),
            polymul(y1, x2, config.q, config.croot),
            config.q,
            config.croot
        ) * config.t) / config.q
    c3n = (polymul(y1, y2, config.q, config.croot) * config.t) / config.q

    c1 = np.round(c1n).astype(np.int64) % config.q
    c2 = np.round(c2n).astype(np.int64)% config.q
    c3 = np.round(c3n).astype(np.int64) % config.q

    return (c1, c2, c3)

def relin(c1, c2, c3, eval):
    eval, a = eval

    c1_hat = polymul(eval, c3, config.q, config.croot)
    c1_hat = polyadd(c1_hat, c1, config.q, config.croot)

    c2_hat = polymul(a, c3, config.q, config.croot)
    c2_hat = polyadd(c2_hat, c2, config.q, config.croot)

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
    b = polyadd(polymul(-a, sk, config.q, config.croot), -e, config.q, config.croot)

    # Eval
    eval = polymul(a, sk, config.q, config.croot)
    eval = polyadd(eval, e, config.q, config.croot)
    eval = -1 * eval
    eval = poly.polydiv(eval % config.q, config.croot)[1] % config.q
    sk2 = polymul(sk, sk, config.q, config.croot)
    eval = polyadd(eval, sk2, config.q, config.croot)

    return (b, a), sk, (eval, a)


if __name__ == '__main__':

    d = 3

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
    logger.info(f"Decoding:\n{int(Polynomial(plain)(2))}\n")

    sum = add(cipher, cipher).decrypt(sk)
    
    # Decode sum
    logger.info(f"Decoding:\n{int(Polynomial(sum)(2))}\n")

    prod = mul(cipher, cipher)
    prod = relin(*prod, eval).decrypt(sk)

    logger.info(f"Decoding:\n{int(Polynomial(prod)(2))}\n")

