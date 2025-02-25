import numpy as np

# 1. Triangular membership function
def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:  # b <= x < c
        return (c - x) / (c - b)

# 2. Define fuzzy sets
quality_sets = {
    "poor":    (0, 0, 5),
    "average": (0, 5, 10),
    "good":    (5, 10, 10)
}

service_sets = {
    "poor":    (0, 0, 5),
    "average": (0, 5, 10),
    "good":    (5, 10, 10)
}

tip_sets = {
    "low":    (0, 0, 13),
    "medium": (0, 13, 25),
    "high":   (13, 25, 25)
}

# 3. Fuzzification function
def fuzzify(value, fuzzy_sets):
    memberships = {}
    for label, (a, b, c) in fuzzy_sets.items():
        memberships[label] = trimf(value, a, b, c)
    return memberships

# 4. Define rules (for Mamdani inference)
rules = [
    ([("quality", "poor"), ("service", "poor")],        "AND", ("tip", "low")),
    ([("quality", "average"), ("service", "average")],  "OR",  ("tip", "medium")),
    ([("quality", "good"), ("service", "good")],        "AND",  ("tip", "high")),
]

# 5. Inference (build aggregated output)
def mamdani_inference(quality_mf, service_mf, rules, tip_sets, num_points=101):
    tip_range = np.linspace(0, 25, num_points)
    aggregated = np.zeros_like(tip_range)
    
    for antecedents, operator, consequent in rules:
        # 1. Compute rule firing strength
        degrees = []
        for (var_name, set_label) in antecedents:
            if var_name == "quality":
                deg = quality_mf[set_label]
            elif var_name == "service":
                deg = service_mf[set_label]
            else:
                raise ValueError("Unknown variable in rule antecedent.")
            degrees.append(deg)
        
        if operator.upper() == "AND":
            alpha = min(degrees)
        elif operator.upper() == "OR":
            alpha = max(degrees)
        else:
            raise ValueError("Operator must be 'AND' or 'OR'.")
        
        # 2. Clip consequent membership
        _, tip_label = consequent
        a, b, c = tip_sets[tip_label]
        consequent_mf = np.array([trimf(x, a, b, c) for x in tip_range])
        clipped_mf = np.minimum(consequent_mf, alpha)
        
        # 3. Aggregate (union = max)
        aggregated = np.maximum(aggregated, clipped_mf)
    
    return tip_range, aggregated

# 6. Centroid defuzzification
def centroid_defuzzify(x, mu):
    if np.sum(mu) == 0:
        return 0.0
    return np.sum(x * mu) / np.sum(mu)

# 7. Example usage
if __name__ == "__main__":
    # Crisp inputs
    quality_value = 0.01
    service_value = 0.01
    
    # Fuzzify inputs
    quality_mf = fuzzify(quality_value, quality_sets)
    service_mf = fuzzify(service_value, service_sets)
    
    # Inference
    tip_range, aggregated = mamdani_inference(quality_mf, service_mf, rules, tip_sets)
    
    # Defuzzify
    tip_value = centroid_defuzzify(tip_range, aggregated)
    
    print(f"Quality = {quality_value}, Service = {service_value}")
    print(f"Suggested Tip = {tip_value:.2f}%")
