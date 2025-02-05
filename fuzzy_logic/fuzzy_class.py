import numpy as np

class FuzzyLogicSystem:
    """
    A simple fuzzy logic system that demonstrates:
     - Common membership functions
     - Storage of fuzzy sets for input and output variables
     - A simple method to compute membership values (fuzzification)
     - An example method to compute a single fuzzy output from membership values
    """
    
    def __init__(self):
        self.input_variables = {}
        self.output_variables = {}

    # ===== Membership Functions =====
    
    @staticmethod
    def triangular_mf(x, a, b, c):
        """
        Triangular membership function.
        Parameters:
            x (float): Crisp input
            a, b, c (float): Points of the triangle (with a < b < c)
        Returns:
            float: Membership degree of x in [0, 1]
        """
        if x <= a or x >= c:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        else:  # b <= x < c
            return (c - x) / (c - b)

    @staticmethod
    def trapezoidal_mf(x, a, b, c, d):
        """
        Trapezoidal membership function.
        Parameters:
            x (float): Crisp input
            a, b, c, d (float): Points of the trapezoid (a < b <= c < d)
        Returns:
            float: Membership degree of x in [0, 1]
        """
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c)

    @staticmethod
    def gaussian_mf(x, mean, sigma):
        """
        Gaussian membership function.
        Parameters:
            x (float): Crisp input
            mean (float): Center of the Gaussian
            sigma (float): Standard deviation
        Returns:
            float: Membership degree of x in [0, 1]
        """
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    # ===== Classes to store fuzzy sets and variables =====
    
    class FuzzySet:
        """
        A container for one fuzzy set, which has:
         - A label (e.g., 'Cold', 'Warm', 'Hot')
         - A membership function (callable)
         - The parameters for that membership function
        """
        def __init__(self, label, mf_callable, params):
            self.label = label
            self.mf_callable = mf_callable
            self.params = params

        def membership(self, x):
            """Compute membership degree for a crisp value x."""
            return self.mf_callable(x, *self.params)

    class FuzzyVariable:
        """
        Represents a fuzzy variable, which can be:
         - An input variable (e.g., Temperature)
         - An output variable (e.g., FanSpeed)
        It contains multiple fuzzy sets (e.g., 'Cold', 'Warm', 'Hot').
        """
        def __init__(self, name, range_min, range_max):
            self.name = name
            self.range_min = range_min
            self.range_max = range_max
            self.fuzzy_sets = {}

        def add_fuzzy_set(self, fuzzy_set):
            """Add a fuzzy set to this variable."""
            self.fuzzy_sets[fuzzy_set.label] = fuzzy_set

        def fuzzify(self, x):
            """
            Compute the membership values of crisp input x for each fuzzy set.
            Returns a dictionary: {fuzzy_set_label: membership_value}
            """
            memberships = {}
            for label, fset in self.fuzzy_sets.items():
                memberships[label] = fset.membership(x)
            return memberships

    # ===== Utility functions =====

    def add_input_variable(self, fuzzy_variable):
        """Add a fuzzy input variable (e.g., Temperature)."""
        self.input_variables[fuzzy_variable.name] = fuzzy_variable

    def add_output_variable(self, fuzzy_variable):
        """Add a fuzzy output variable (e.g., FanSpeed)."""
        self.output_variables[fuzzy_variable.name] = fuzzy_variable

    def compute_fuzzy_output(self, input_name, input_value, output_name):
        """
        Simple demonstration: 
         1) Fuzzify the input value.
         2) Combine membership functions to produce a single fuzzy output.
            (Here we just take the max membership across sets for demonstration.)
         3) Return a dictionary of output membership values.
        """
        if input_name not in self.input_variables:
            raise ValueError(f"Unknown input variable: {input_name}")
        if output_name not in self.output_variables:
            raise ValueError(f"Unknown output variable: {output_name}")

        # Fuzzify the input
        input_fv = self.input_variables[input_name]
        input_memberships = input_fv.fuzzify(input_value)

        # For simplicity, assume output membership is just the same labels
        # and we take the maximum membership value. 
        # In a real system, you would have rules to map input fuzzy sets to output fuzzy sets.
        max_membership = max(input_memberships.values()) if input_memberships else 0.0

        # Then for the output, let's just set all sets to the same membership 
        # (again, purely for demonstration)
        output_fv = self.output_variables[output_name]
        output_memberships = {}
        for label in output_fv.fuzzy_sets:
            output_memberships[label] = max_membership

        return output_memberships


# ===== Example Usage =====

if __name__ == "__main__":
    # Create a fuzzy logic system
    fls = FuzzyLogicSystem()

    # Create an input variable: Temperature in the range [0, 40]
    temp_var = fls.FuzzyVariable("Temperature", 0, 40)

    # Add fuzzy sets to "Temperature"
    # Example sets: COLD (triangular: 0,0,20), WARM (triangular: 10,20,30), HOT (triangular: 20,40,40)
    cold_set = fls.FuzzySet("COLD", fls.triangular_mf, (0, 0, 20))
    warm_set = fls.FuzzySet("WARM", fls.triangular_mf, (10, 20, 30))
    hot_set  = fls.FuzzySet("HOT",  fls.triangular_mf, (20, 40, 40))
    
    temp_var.add_fuzzy_set(cold_set)
    temp_var.add_fuzzy_set(warm_set)
    temp_var.add_fuzzy_set(hot_set)

    # Register the input variable with the system
    fls.add_input_variable(temp_var)

    # Create an output variable: FanSpeed in the range [0, 1]
    fan_var = fls.FuzzyVariable("FanSpeed", 0, 1)

    # Add fuzzy sets to "FanSpeed"
    # Example sets: LOW (trapezoidal: 0,0,0.3,0.5), MEDIUM (triangular: 0.3,0.5,0.7), HIGH (trapezoidal: 0.5,0.7,1.0,1.0)
    low_set = fls.FuzzySet("LOW", fls.trapezoidal_mf, (0.0, 0.0, 0.3, 0.5))
    med_set = fls.FuzzySet("MEDIUM", fls.triangular_mf, (0.3, 0.5, 0.7))
    high_set = fls.FuzzySet("HIGH", fls.trapezoidal_mf, (0.5, 0.7, 1.0, 1.0))
    
    fan_var.add_fuzzy_set(low_set)
    fan_var.add_fuzzy_set(med_set)
    fan_var.add_fuzzy_set(high_set)

    # Register the output variable with the system
    fls.add_output_variable(fan_var)

    # ===== Fuzzify an example input temperature =====
    example_temp = 1
    print(f"Fuzzification of Temperature = {example_temp}°C:")
    memberships = temp_var.fuzzify(example_temp)
    for label, m_value in memberships.items():
        print(f"  {label}: {m_value:.3f}")

    # ===== Compute a simple fuzzy output from that input =====
    fuzzy_output = fls.compute_fuzzy_output("Temperature", example_temp, "FanSpeed")
    print("\nResulting FanSpeed fuzzy output (demonstration):")
    for label, m_value in fuzzy_output.items():
        print(f"  {label}: {m_value:.3f}")
