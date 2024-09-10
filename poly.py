import numpy as np

class Polynomial:

    def __init__(self, coef, mod=True, negacyclic=False):
        """
        Initialize a Polynomial.

        args:
            coef: (list or np.ndarray) Polynomial coefficients
        """
        if isinstance(coef, list):
            self.coef = np.array(coef)
        elif isinstance(coef, np.ndarray):
            assert coef.ndim == 1, "Coefficients must be one-dimensional"
            self.coef = coef
        else:
            raise ValueError("Coefficients must be list or numpy array")

        self.mod = 7681 # Coefficient modulo
        self.n = 4
        self.w = self.find_primitive_root(self.n, self.mod)
        self.w_inv = self.modular_inverse(self.w, self.mod)
        self.n_inv = self.modular_inverse(self.n, self.mod)
        self.w2 = self.find_2nth_primitive_root(self.n, self.mod)
        self.w2_inv = self.modular_inverse(self.w2, self.mod)
        if mod:
            self.root = Polynomial([-1, 0, 0, 0, 1], mod=False)
        if negacyclic:
            self.root[0] = 1

    
    def __mul__(self, other):
        """
        Multiply operator overload.

        arg:
            other: (Polynomial) The RHS factor.

        returns:
            poly: (Polynomial) The product of LHS and RHS.
        """
        poly = self.ntt_neg_multiply(self.coef, other.coef)
        poly = poly % self.mod % self.root
        return poly

    
    def __mod__(self, other):
        """
        Modulo operator overload.

        args:
            other: (int or Polynomial): If integer, modulo the coefficients.
            If Polynomial, modulo the degree.

        returns:
            poly: (Polynomial): The modulated polynomial.
        """
        if isinstance(other, Polynomial):
            _, remainder = np.polydiv(
                np.trim_zeros(self.coef[::-1], trim='f'),  
                np.trim_zeros(other.coef[::-1], trim='f')
            )
            return Polynomial(remainder[::-1])
        elif isinstance(other, int):
            return Polynomial(self.coef % other)
        else:
            raise ValueError("Modulo operator must be poly or int")

    
    def discrete_linear_conv(self, lhs, rhs):
        """
        Multiply polynomials using a discrete linear convolution.

        args:
            lhs: (array) lhs coefficients
            rhs: (array) rhs coefficients

        returns:
            poly: (Polynomial) Coefficients of the polynomial product.
        """

        return Polynomial(np.convolve(lhs, rhs))


    def find_primitive_root(self, n, q):
        """
        Find a primitive n-th root of unity in the finite field f_q.

        Args:
            n (int): The order of the root of unity.
            q (int): The size of the field, which should be a prime number p.

        Returns:
            int: A primitive n-th root of unity in f_q.

        Raises:
            ValueError: If primitive root cannot be found.
        """

        # Check that n divides q-1 (required)
        if (q - 1) % n != 0:
            return None

        def is_primitive_root(w):
            if pow(w, n, q) != 1:
                return False
            for k in range(1, n):
                if pow(w, k, q) == 1:
                    return False
            return True

        for w in range(1, q):
            if is_primitive_root(w):
                return w
        
        raise valueerror(f"N-th primitive root does not exist for {n} modulo {q}")

    
    def find_2nth_primitive_root(self, n, q):
        """
        Find a primitive 2n-th root of unity in the finite field f_q.

        Args:
            n (int): The order of the root of unity.
            q (int): The size of the field, which should be a prime number p.

        Returns:
            int: A primitive 2n-th root of unity in f_q.

        Raise:
            ValueError: If primitive 2nth root cannot be found.
        """
        w = self.find_primitive_root(n, q)

        def is_2nth_primitive_root(v):
            if pow(v, 2, q) != w:
                return False
            if pow(v, n, q) != -1 % q:
                return False
            return True

        for v in range(1,q):
            if is_2nth_primitive_root(v):
                return v

        raise valueerror(f"2n-th primitive root does not exist for {n} modulo {q}")


    def modular_inverse(self, a, p):
        """
        Compute the modular inverse of a modulo p using the
        Extended Euclidean Algorithm.
        
        Args:
            a (int): The number to find the inverse of.
            p (int): The modulus, which should be a prime number.
        
        Returns:
            int: The modular inverse of a modulo p.

        Raises:
            ValueError: If no modular inverse exists.
        """
        def extended_gcd(a, b):
            if a == 0:
                return (b, 0, 1)
            g, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return (g, x, y)
        
        g, x, y = extended_gcd(a, p)
        if g != 1:
            raise ValueError(f"No modular inverse exists for {a} modulo {p}")
        else:
            return x % p


    def ntt_pos_multiply(self, lhs, rhs):
        """
        Performs polynomial multiplication using positive-wrapped
        number theoretic transform.

        Args:
            lhs: (array) lhs coefficients
            rhs: (array) rhs coefficients

        Returns:
            poly: (Polynomial) Coefficients of polynomial product.
        """
        lhs = self.ntt_pos(lhs)
        rhs = self.ntt_pos(rhs)
        return Polynomial(self.intt_pos(lhs * rhs) % self.mod)


    def ntt_neg_multiply(self, lhs, rhs):
        """
        Performs polynomial multiplication using negative-wrapped
        number theoretic transform.

        Args:
            lhs: (array) lhs coefficients
            rhs: (array) rhs coefficients

        Returns:
            poly: (Polynomial) Coefficients of polynomial product.
        """
        lhs = self.ntt_neg(lhs)
        rhs = self.ntt_neg(rhs)
        return Polynomial(self.intt_neg(lhs * rhs) % self.mod)


    def ntt_pos(self, coef):
        """
        Calculates the positive-wrapped number theoretic
        transform of a polynomial.

        Args:
            coef: (array) Polynomial coefficients

        Returns:
            (array) Coefficients of NTT transformation
        """
        w_matrix = np.arange(self.n)[:, None] * np.arange(self.n)
        w_matrix = w_matrix % self.n
        w_transform = np.vectorize(lambda x: (self.w ** x) % self.mod)
        w_matrix = w_transform(w_matrix)
        return np.dot(w_matrix, coef) % self.mod


    def intt_pos(self, coef):
        """
        Calculates the positive-wrapped number theoretic
        transform inverse of a polynomial.

        Args:
            coef: (array) Polynomial coefficients

        Returns:
            (array) Coefficients from the original transformation
        """
        w_matrix = np.arange(self.n)[:, None] * np.arange(self.n)
        w_matrix = w_matrix % self.n
        w_transform = np.vectorize(lambda x: (self.w_inv ** x) % self.mod)
        w_matrix = w_transform(w_matrix)
        return np.dot(self.n_inv * w_matrix, coef) % self.mod


    def ntt_neg(self, coef):
        """
        Calculates the negative-wrapped number theoretic
        transform of a polynomial.

        Args:
            coef: (array) Polynomial coefficients

        Returns:
            (array) Coefficients of NTT transformation
        """
        w_matrix = np.arange(self.n)[:, None] * np.arange(self.n)
        w_matrix = w_matrix * 2 + np.arange(self.n)
        w_matrix = w_matrix % (self.n * 2)
        w_transform = np.vectorize(lambda x: (self.w2 ** x) % self.mod)
        w_matrix = w_transform(w_matrix)
        return np.dot(w_matrix, coef) % self.mod


    def intt_neg(self, coef):
        """
        Calculates the negative-wrapped number theoretic
        transform inverse of a polynomial.

        Args:
            coef: (array) Polynomial coefficients

        Returns:
            (array) Coefficients from the original transformation
        """
        w_matrix = np.arange(self.n)[:, None] * np.arange(self.n)
        w_matrix = w_matrix * 2 + np.arange(self.n)[:, None]
        w_matrix = w_matrix % (self.n * 2)
        w_transform = np.vectorize(lambda x: (self.w2_inv ** x) % self.mod)
        w_matrix = w_transform(w_matrix)
        return np.dot(self.n_inv * w_matrix, coef) % self.mod


    def fast_ntt(self):

        coef = np.zeros(self.n)

        for j in range(self.n // 2):

            Aj = 0
            Bj = 0

            for i in range(self.n // 2):
                
                # O(n**2)
                #Aj += pow(self.w2, 4*i*j+2*i, self.mod) * self.coef[2*i]
                #Bj += pow(self.w2, 4*i*j+2*i, self.mod) * self.coef[2*i+1]

                # O(nlogn)
                Aj, Bj = self.ntt_helper(0, self.n // 2, j)

            coef[j] = Aj + pow(self.w2, 2*j+1, self.mod) * Bj
            coef[j+self.n//2] = Aj - pow(self.w2, 2*j+1, self.mod) * Bj

            coef[j] = coef[j] % self.mod
            coef[j+self.n//2] = coef[j+self.n//2] % self.mod

        return coef

    def ntt_helper(self, start, end, j):
        
        # Base case
        if start >= end:
            return (0, 0)

        # Processing case
        if start + 1 == end:
            i = start
            pow_value = pow(self.w2, 4*i*j + 2*i, self.mod)
            return (self.coef[2*i] * pow_value, self.coef[2*i+1] * pow_value)

        # Recurssive case
        mid = (start + end) // 2
        Aj1, Bj1 = self.ntt_helper(start, mid, j)
        Aj2, Bj2 = self.ntt_helper(mid, end, j)

        # Combine results
        Aj = Aj1 + Aj2
        Bj = Bj1 + Bj2

        return (Aj, Bj)


    def __str__(self):
        """
        Pretty print.
        """
        superscript = {
            0: '⁰',
            1: '¹',
            2: '²',
            3: '³', 
            4: '⁴',
            5: '⁵',
            6: '⁶',
            7: '⁷',
            8: '⁸',
            9: '⁹'
        }

        s = ""
        
        for pow, coef in enumerate(self.coef):
            if coef != 0:
                sign = "+" if coef > 0 else "-"
                exponent = "".join(superscript[int(num)] for num in str(pow))
                s+=f"{sign} {abs(coef)}x{exponent} "
        
        # Remove leading sign if positive
        if s:
            s = s[1:] if s[0] == "+" else s
            s = s.strip()
            return s
        else:
            return "0"
            




        
