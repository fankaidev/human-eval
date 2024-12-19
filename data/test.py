def is_multiply_prime(a):
    """Write a function that returns true if the given number is the multiplication of 3 prime numbers
    and false otherwise.
    Knowing that (a) is less then 100.
    Example:
    is_multiply_prime(30) == True
    30 = 2 * 3 * 5
    """

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    for i in range(2, a):
        if a % i == 0 and is_prime(i):
            for j in range(i, a // i):
                if a % j == 0 and is_prime(j):
                    k = a // (i * j)
                    if i * j * k == a and is_prime(k):
                        print(i, j, k)
                        return True
    return False


def test_is_multiply_prime():
    # assert is_multiply_prime(5) == False
    # assert is_multiply_prime(30) == True
    # assert is_multiply_prime(8) == True
    assert is_multiply_prime(10) == False
    assert is_multiply_prime(125) == True
    assert is_multiply_prime(3 * 5 * 7) == True
    assert is_multiply_prime(3 * 6 * 7) == False
    assert is_multiply_prime(9 * 9 * 9) == False
    assert is_multiply_prime(11 * 9 * 9) == False
    assert is_multiply_prime(11 * 13 * 7) == True

    print("All test cases passed!")


if __name__ == "__main__":
    test_is_multiply_prime()
