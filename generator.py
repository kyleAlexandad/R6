import random
from itertools import combinations_with_replacement

def generate_random_polynomials(n, m, C, num_polynomials=10000):

    all_monomials = []
    for total_degree in range(n + 1):
        for combo in combinations_with_replacement(range(m), total_degree):
            exponents = [0] * m
            for var_idx in combo:
                exponents[var_idx] += 1
            
            all_monomials.append(tuple(exponents))
    
    all_monomials.sort()

    # indexed list
    monomial_to_index = {monomial: idx for idx, monomial in enumerate(all_monomials)}

    all_polynomials = []
    
    for _ in range(num_polynomials):
        poly = generate_random_polynomial(n, m, C)
        
        # Convert to vector
        vector = [0] * len(all_monomials)
        for monomial, coef in poly.items():
            if monomial in monomial_to_index:
                vector[monomial_to_index[monomial]] = coef
        
        all_polynomials.append(vector)
    
    return all_polynomials

def generate_random_polynomial(n, m, C):

    if random.choice([True, False]):
        # Start with 1
        poly = {(0,) * m: 1}
    else:
        # Start with a single variable
        var_idx = random.randint(0, m - 1)
        exponents = [0] * m
        exponents[var_idx] = 1
        poly = {tuple(exponents): 1}
    
    # Perform C successive operations
    for _ in range(C):

        # Choose between addition and multiplication
        operation = random.choice(["add", "multiply"])
        
        # Generate a simple term
        if random.choice([True, False]):
            # Use 1
            new_poly = {(0,) * m: 1}
        else:
            # Use a single variable
            var_idx = random.randint(0, m - 1)
            exponents = [0] * m
            exponents[var_idx] = 1
            new_poly = {tuple(exponents): 1}
        
        if operation == "add":
            poly = add_polynomials(poly, new_poly, n)
        else:  # multiply
            poly = multiply_polynomials(poly, new_poly, n)
    
    # print(f"poly:  {poly}")
    return poly

def add_polynomials(poly1, poly2, n):
    result = poly1.copy()
    
    for monomial, coef in poly2.items():
        if sum(monomial) > n:
            continue  # Skip terms with degree > n
        
        if monomial in result:
            result[monomial] = (result[monomial] + coef) % 2
            if result[monomial] == 0:
                del result[monomial]
        else:
            result[monomial] = coef
    
    return result

def multiply_polynomials(poly1, poly2, n):
    result = {}
    
    for mon1, coef1 in poly1.items():
        for mon2, coef2 in poly2.items():
            # Add the exponents
            new_mon = tuple(e1 + e2 for e1, e2 in zip(mon1, mon2))
            
            # Skip terms with degree > n
            if sum(new_mon) > n:
                continue
            
            # Multiply the coefficients (both are 1, so result is 1)
            new_coef = coef1 * coef2  # This will always be 1
            
            # Add to the result
            if new_mon in result:
                result[new_mon] = (result[new_mon] + new_coef) % 2
                if result[new_mon] == 0:
                    del result[new_mon]
            else:
                result[new_mon] = new_coef
    
    return result

polyn = generate_random_polynomials(5,6,4)

print(polyn)
