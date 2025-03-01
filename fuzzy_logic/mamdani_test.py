import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import time

def trimf(x, a, b, c):
    """
    Triangular membership function.
    """
    eps = 1e-8
    return np.maximum(np.minimum((x - a) / (b - a + eps), (c - x) / (c - b + eps)), 0)

def trapmf(x, a, b, c, d):
    """
    Trapezoidal membership function.
    """
    eps = 1e-8
    return np.maximum(np.minimum(np.minimum((x - a) / (b - a + eps), 1), (d - x) / (d - c + eps)), 0)

class MamdaniFIS:
    def __init__(self, output_range=(0, 1), n_points=1000):
        # Create the output universe for defuzzification.
        self.n_points = n_points
        self.output_universe = np.linspace(output_range[0], output_range[1], n_points)
        # Dictionaries to hold membership functions.
        self.input_mfs = {}   # e.g., { 'x': {'mf_0': func, 'mf_1': func, ...}, ... }
        self.output_mfs = {}  # e.g., { 'mf_0': func, 'mf_1': func, ... }
        self.rules = []       # List of fuzzy rules.

    def _generate_membership_parameters(self, centers, domain=(0,1), shoulder_frac=0.5):
        """
        Given a list of centers (which must lie strictly between the domain boundaries),
        generate a list of membership function definitions so that:
         - The left extreme MF is a left shoulder defined as a trapezoid:
              parameters: (domain_start, domain_start, L_top, first_center)
         - The interior MFs are triangles whose bases touch the shoulder tops:
              left base of first interior = L_top, right base of last interior = R_top.
         - The right extreme MF is a right shoulder defined as a trapezoid:
              parameters: (last_center, R_top, domain_end, domain_end)
              
        Here, L_top = domain_start + shoulder_frac*(first_center - domain_start)  
        and   R_top = domain_end - shoulder_frac*(domain_end - last_center)
        
        For interior centers (if more than one), use standard midpoints.
        
        Returns a list of tuples: (mf_type, parameters)
          - For trapezoidal MFs: ("trap", (a, b, c, d))
          - For triangular MFs: ("tri", (a, b, c))
        """
        centers = sorted(centers)
        L, R = domain
        params_list = []
        
        # Left extreme: trapezoidal MF.
        L_top = L + shoulder_frac * (centers[0] - L)
        left_trap = ("trap", (L, L, L_top, centers[0]))
        params_list.append(left_trap)
        
        n = len(centers)
        if n == 1:
            # Only one given center: define one interior MF.
            R_top = R - shoulder_frac * (R - centers[0])
            interior = ("tri", (L_top, centers[0], R_top))
            params_list.append(interior)
        else:
            # For the first interior MF:
            next_mid = centers[1]
            interior_first = ("tri", (L_top, centers[0], next_mid))
            params_list.append(interior_first)
            # For additional interior MFs, if any.
            for i in range(1, n-1):
                left_base = centers[i-1]
                right_base = centers[i+1]
                interior = ("tri", (left_base, centers[i], right_base))
                params_list.append(interior)
            # For the last interior MF:
            R_top = R - shoulder_frac * (R - centers[-1])
            interior_last = ("tri", ((centers[-2]), centers[-1], R_top))
            params_list.append(interior_last)
        
        # Right extreme: trapezoidal MF.
        R_top = R - shoulder_frac * (R - centers[-1])
        right_trap = ("trap", (centers[-1], R_top, R, R))
        params_list.append(right_trap)
        
        return params_list

    def add_input_triangles(self, var, centers, domain=(0,1), shoulder_frac=0.5):
        """
        Automatically generate and add membership functions for an input variable using the given centers.
        Extreme sets are defined as trapezoidal shoulders and interior sets as triangles.
        """
        params_list = self._generate_membership_parameters(centers, domain, shoulder_frac)
        if var not in self.input_mfs:
            self.input_mfs[var] = {}
        for i, (mf_type, params) in enumerate(params_list):
            label = f"mf_{i}"
            if mf_type == "tri":
                self.input_mfs[var][label] = lambda x, a=params[0], b=params[1], c=params[2]: trimf(x, a, b, c)
            else:  # "trap"
                self.input_mfs[var][label] = lambda x, a=params[0], b=params[1], c=params[2], d=params[3]: trapmf(x, a, b, c, d)

    def add_output_triangles(self, centers, domain=(0,1), shoulder_frac=0.5):
        """
        Automatically generate and add membership functions for the output variable.
        """
        params_list = self._generate_membership_parameters(centers, domain, shoulder_frac)
        for i, (mf_type, params) in enumerate(params_list):
            label = f"mf_{i}"
            if mf_type == "tri":
                self.output_mfs[label] = lambda x, a=params[0], b=params[1], c=params[2]: trimf(x, a, b, c)
            else:
                self.output_mfs[label] = lambda x, a=params[0], b=params[1], c=params[2], d=params[3]: trapmf(x, a, b, c, d)

    def add_input_mf(self, var, label, mf_func):
        if var not in self.input_mfs:
            self.input_mfs[var] = {}
        self.input_mfs[var][label] = mf_func

    def add_output_mf(self, label, mf_func):
        self.output_mfs[label] = mf_func

    def add_rule(self, antecedents, output_label):
        """
        Add a fuzzy rule.
        antecedents: dict mapping variable names to MF labels.
        output_label: label of the output MF.
        """
        self.rules.append((antecedents, output_label))

    def infer(self, inputs):
        """
        Perform fuzzy inference using centroid averaging.
        """
        weighted_sum = 0.0
        total_area = 0.0
        for antecedents, output_label in self.rules:
            degrees = []
            for var, label in antecedents.items():
                mf_func = self.input_mfs[var][label]
                degrees.append(mf_func(inputs[var]))
            firing_strength = np.min(degrees)
            output_values = self.output_mfs[output_label](self.output_universe)
            clipped = np.fmin(firing_strength, output_values)
            area = np.sum(clipped)
            if area > 0:
                centroid = np.sum(clipped * self.output_universe) / area
            else:
                centroid = 0
            weighted_sum += centroid * area
            total_area += area
        if total_area == 0:
            return 0
        return weighted_sum / total_area

    def plot_response_surface(self, x_range, y_range, x_points=50, y_points=50):
        x_vals = np.linspace(x_range[0], x_range[1], x_points)
        y_vals = np.linspace(y_range[0], y_range[1], y_points)
        max_val = -100000
        min_val = 100000
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        for i in range(x_points):
            for j in range(y_points):
                inp = {'x': X[j, i], 'y': Y[j, i]}
                Z[j, i] = self.infer(inp)
                if Z[j,i] > max_val:
                    max_val = Z[j,i]
                if Z[j,i] < min_val:
                    min_val = Z[j,i]
        print("Max value:", max_val)
        print("Min value:", min_val)
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_xlabel('Input x')
        ax.set_ylabel('Input y')
        ax.set_zlabel('Output z')
        ax.set_title("FIS Response Surface")
        fig.colorbar(surface, shrink=0.5, aspect=5)
        plt.show()

    def plot_memberships(self, domain=(0,1), num_points=1000):
        x_vals = np.linspace(domain[0], domain[1], num_points)
        # Plot membership functions for each input variable.
        for var, mf_dict in self.input_mfs.items():
            plt.figure()
            for label, mf in mf_dict.items():
                plt.plot(x_vals, mf(x_vals), label=label)
            plt.title(f"Membership Functions for Input '{var}'")
            plt.xlabel("Domain")
            plt.ylabel("Membership Degree")
            plt.legend()
            plt.grid(True)
            plt.show()
        # Plot membership functions for the output.
        if self.output_mfs:
            plt.figure()
            for label, mf in self.output_mfs.items():
                plt.plot(x_vals, mf(x_vals), label=label)
            plt.title("Membership Functions for Output")
            plt.xlabel("Domain")
            plt.ylabel("Membership Degree")
            plt.legend()
            plt.grid(True)
            plt.show()

# --- Example usage ---
if __name__ == "__main__":
    # Create a FIS whose output is in [0, 1].
    fis = MamdaniFIS(output_range=(0,1), n_points=1000)
    
    # For input variables, use one center (0.5) so that the MFs become:
    # Left shoulder, interior MF, and right shoulder.
    fis.add_input_triangles("x", [0.5], domain=(0,1), shoulder_frac=0.1)
    fis.add_input_triangles("y", [0.5], domain=(0,1), shoulder_frac=0.1)
    
    # For the output, choose a center (e.g., 0.25) to force low output.
    fis.add_output_triangles([0.25,0.5,0.75], domain=(0,1), shoulder_frac=0.003)
    
    # Define a simple rule base.
    fis.add_rule({'x': 'mf_0', 'y': 'mf_0'}, 'mf_2')
    fis.add_rule({'x': 'mf_1', 'y': 'mf_0'}, 'mf_1')
    fis.add_rule({'x': 'mf_2', 'y': 'mf_0'}, 'mf_0')

    fis.add_rule({'x': 'mf_0', 'y': 'mf_1'}, 'mf_2')
    fis.add_rule({'x': 'mf_1', 'y': 'mf_1'}, 'mf_3')
    fis.add_rule({'x': 'mf_2', 'y': 'mf_1'}, 'mf_4')

    fis.add_rule({'x': 'mf_0', 'y': 'mf_2'}, 'mf_2')
    fis.add_rule({'x': 'mf_1', 'y': 'mf_2'}, 'mf_1')
    fis.add_rule({'x': 'mf_2', 'y': 'mf_2'}, 'mf_0')
    
    # For x nearly high (0.99) and y low (0.25), we expect a low output.
    result = fis.infer({'x': 0.99, 'y': 0.25})
    print("Inference result for x=0.99, y=0.25:", result)
    
    # Plot the membership functions and the response surface.
    fis.plot_memberships(domain=(0.000001,0.999999), num_points=1000)
    fis.plot_response_surface(x_range=(0.000001,0.999999), y_range=(0.000001,0.999999), x_points=20, y_points=20)
