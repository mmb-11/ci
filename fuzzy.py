def fuzzy_union(A, B):
    return {x: max(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_intersection(A, B):
    return {x: min(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_complement(A):
    return {x: round(1 - A[x], 2) for x in A}

def fuzzy_difference(A, B):
    return {x: min(A.get(x, 0), 1 - B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_cartesian_product(A, B):
    return {(x, y): min(A[x], B[y]) for x in A for y in B}

def max_min_composition(R, S):
    result = {}
    for (x, y1) in R:
        for (y2, z) in S:
            if y1 == y2:
                key = (x, z)
                value = min(R[(x, y1)], S[(y2, z)])
                if key in result:
                    result[key] = max(result[key], value)
                else:
                    result[key] = value
    return result

#Fuzzy sets
A = {'x1': 0.3, 'x2': 0.7}
B = {'y1': 0.6, 'y2': 0.4}

#Fuzzy relations
R = {('x1', 'y1'): 0.3, ('x1', 'y2'): 0.7}
S = {('y1', 'z1'): 0.5, ('y2', 'z1'): 0.8}

print("Fuzzy Union:", fuzzy_union(A, B))
print("\nFuzzy Intersection:", fuzzy_intersection(A, B))
print("\nFuzzy Complement of A:", fuzzy_complement(A))
print("\nFuzzy Difference (A - B):", fuzzy_difference(A, B))
print("\nFuzzy Cartesian Product (A x B):", fuzzy_cartesian_product(A, B))
print("\nMax-Min Composition (R ∘ S):", max_min_composition(R, S))
