# Pi Cruncher Benchmark
#
# This script calculates Pi to a specified number of decimal places using the
# Gauss-Legendre algorithm. It's designed to be a CPU-intensive task,
# making it useful as a simple benchmark to stress-test a processor and
# measure its power draw under a heavy, sustained load.

import decimal
import time
import sys


def calculate_pi(digits):
    """
    Calculates Pi to a specified number of digits using the Gauss-Legendre algorithm.
    """
    if digits <= 0:
        raise ValueError("Number of digits must be a positive integer.")

    # Set the precision for the decimal module.
    # We need a few extra digits for precision during calculation.
    decimal.getcontext().prec = digits + 3

    # Initialize algorithm constants
    a = decimal.Decimal(1)
    b = decimal.Decimal(1) / decimal.Decimal(2).sqrt()
    t = decimal.Decimal(1) / decimal.Decimal(4)
    p = decimal.Decimal(1)

    a_old = a

    print(f"Starting Pi calculation to {digits} digits...")
    print("This will put a heavy load on your CPU.")

    iteration = 0
    while True:
        iteration += 1
        a_new = (a + b) / 2
        b = (a * b).sqrt()
        t -= p * (a - a_new) ** 2
        a = a_new
        p *= 2

        # Check for convergence
        if a == b:
            break

        # Optional: uncomment the line below to see progress per iteration
        # print(f"Iteration {iteration} complete.")

    pi = (a + b) ** 2 / (4 * t)

    # Return Pi, slicing off the extra precision digits
    return str(pi)[:digits + 1]  # +1 for the "3." at the beginning


def main():
    """
    Main function to parse arguments and run the calculation.
    """
    # Default number of digits if none is provided
    num_digits = 10000

    if len(sys.argv) > 1:
        try:
            num_digits = int(sys.argv[1])
            if num_digits <= 0:
                print("Error: Please provide a positive integer for the number of digits.")
                sys.exit(1)
        except ValueError:
            print("Error: Invalid input. Please enter an integer.")
            sys.exit(1)
    else:
        print(f"No digit count provided. Using default: {num_digits}")

    start_time = time.time()

    pi_result = calculate_pi(num_digits)

    end_time = time.time()

    duration = end_time - start_time

    print("-" * 50)
    print(f"Calculation finished in {duration:.4f} seconds.")
    print(f"Pi to {num_digits} decimal places:")
    # We print only the first and last 50 characters to avoid flooding the console
    if len(pi_result) > 100:
        print(f"{pi_result[:50]}...{pi_result[-50:]}")
    else:
        print(pi_result)
    print("-" * 50)


if __name__ == "__main__":
    main()
