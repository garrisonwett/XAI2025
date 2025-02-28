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
    return np.maximum(np.minimum(np.minimum((x - a) / (b - a + eps), 1),
                                   (d - x) / (d - c + eps)), 0)

class MamdaniFIS:
    def __init__(self, output_range=(0, 1), n_points=1000):
        # Create the output universe for defuzzification.
        self.n_points = n_points
        self.output_universe = np.linspace(output_range[0], output_range[1], n_points)
        # Dictionaries to hold input and output membership functions.
        self.input_mfs = {}   # e.g., { 'x': {'mf_0': func, 'mf_1': func, ...}, ... }
        self.output_mfs = {}  # e.g., { 'mf_0': func, 'mf_1': func, ... }
        # List of fuzzy rules: each is a tuple (antecedents, output_label)
        self.rules = []
    
    @staticmethod
    def _generate_membership_parameters(centers, domain=(0, 1)):
        """
        Given a list of center points (which do NOT include the domain endpoints),
        generate a list of membership function definitions that cover the entire domain.
        
        For the extreme sets (first and last), we use trapezoidal functions so that the
        support exactly reaches the domain boundaries. For the interior sets we use triangles.
        
        Returns a list of tuples: (mf_type, parameters)
          - For interior sets: ("tri", (a, b, c))
          - For extreme sets: ("trap", (a, b, c, d))
          
        Example:
          centers = [0.25, 0.75] with domain (0,1) produces:
             left MF: ("trap", (0, 0, 0.125, 0.25))
             middle MF 1: ("tri", (0, 0.25, 0.75))
             middle MF 2: ("tri", (0.25, 0.75, 1))
             right MF: ("trap", (0.75, 0.875, 1, 1))
        """
        # Include the domain boundaries.
        all_centers = sorted(set([domain[0]] + centers + [domain[1]]))
        mfs = []
        N = len(all_centers)
        for i, peak in enumerate(all_centers):
            if i == 0:
                # Left extreme: use a trapezoidal function that is flat at the left edge.
                mid = (domain[0] + all_centers[1]) / 2
                mfs.append(("trap", (domain[0], domain[0], mid, all_centers[1])))
            elif i == N - 1:
                # Right extreme: use a trapezoidal function flat at the right edge.
                mid = (all_centers[-2] + domain[1]) / 2
                mfs.append(("trap", (all_centers[-2], mid, domain[1], domain[1])))
            else:
                # Interior: use a triangular function.
                mfs.append(("tri", (all_centers[i-1], peak, all_centers[i+1])))
        return mfs

    def add_input_triangles(self, var, centers, domain=(0, 1)):
        """
        Automatically generate and add membership functions for an input variable using
        the provided centers. Extreme sets are defined as trapezoidal, and interior sets
        as triangular. Membership functions are labeled "mf_0", "mf_1", etc.
        """
        mfs = self._generate_membership_parameters(centers, domain)
        if var not in self.input_mfs:
            self.input_mfs[var] = {}
        for i, (mf_type, params) in enumerate(mfs):
            label = f"mf_{i}"
            if mf_type == "tri":
                self.input_mfs[var][label] = lambda x, a=params[0], b=params[1], c=params[2]: trimf(x, a, b, c)
            else:  # trap
                self.input_mfs[var][label] = lambda x, a=params[0], b=params[1], c=params[2], d=params[3]: trapmf(x, a, b, c, d)
    
    def add_output_triangles(self, centers, domain=(0, 1)):
        """
        Automatically generate and add membership functions for the output variable.
        The same approach as for inputs is used.
        """
        mfs = self._generate_membership_parameters(centers, domain)
        for i, (mf_type, params) in enumerate(mfs):
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
        antecedents: dict mapping variable names to membership function labels.
        output_label: label of the output membership function.
        """
        self.rules.append((antecedents, output_label))
    
    def infer(self, inputs):
        """
        Perform fuzzy inference using centroid averaging.
        For each rule, the clipped output MF is defuzzified (centroid computed)
        and then a weighted average is taken.
        """
        weighted_sum = 0.0
        total_area = 0.0
        for antecedents, output_label in self.rules:
            # Determine the rule's firing strength from the input MFs.
            degrees = []
            for var, label in antecedents.items():
                mf_func = self.input_mfs[var][label]
                degrees.append(mf_func(inputs[var]))
            firing_strength = np.min(degrees)
            # Evaluate and clip the output MF.
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
            return 0  # Avoid division by zero.
        return weighted_sum / total_area
    
    def plot_response_surface(self, x_range, y_range, x_points=50, y_points=50):
        """
        Visualize the FIS response surface.
        """
        x_vals = np.linspace(x_range[0], x_range[1], x_points)
        y_vals = np.linspace(y_range[0], y_range[1], y_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        for i in range(x_points):
            for j in range(y_points):
                inp = {'x': X[j, i], 'y': Y[j, i]}
                Z[j, i] = self.infer(inp)
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_xlabel('Input x')
        ax.set_ylabel('Input y')
        ax.set_zlabel('Output z')
        ax.set_title("FIS Response Surface")
        fig.colorbar(surface, shrink=0.5, aspect=5)
        plt.show()
    
    def plot_memberships(self, domain=(0, 1), num_points=1000):
        """
        Plot the membership functions for each input variable and the output.
        """
        x_vals = np.linspace(domain[0], domain[1], num_points)
        # Plot inputs.
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
        # Plot outputs.
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
    # Create a FIS whose output lies in [0,1].
    fis = MamdaniFIS(output_range=(0, 1), n_points=1000)
    
    # For input variables, use a single center so that the MFs split the domain into:
    # left extreme (low), interior (medium), and right extreme (high).
    fis.add_input_triangles("x", [0.5], domain=(0, 1))
    fis.add_input_triangles("y", [0.5], domain=(0, 1))
    
    # For the output, use one center (e.g., 0.25) so that the left extreme output MF covers low values.
    fis.add_output_triangles([0.5], domain=(0, 1))
    
    # Define a rule base.
    # Here, we force that when input y is low (i.e. in its left set 'mf_0'),
    # the output should be low ('mf_0') regardless of x.
    fis.add_rule({'x': 'mf_0', 'y': 'mf_0'}, 'mf_1')
    fis.add_rule({'x': 'mf_1', 'y': 'mf_0'}, 'mf_1')
    fis.add_rule({'x': 'mf_2', 'y': 'mf_0'}, 'mf_1')
    fis.add_rule({'x': 'mf_0', 'y': 'mf_1'}, 'mf_1')
    fis.add_rule({'x': 'mf_1', 'y': 'mf_1'}, 'mf_1')
    fis.add_rule({'x': 'mf_2', 'y': 'mf_1'}, 'mf_1')
    fis.add_rule({'x': 'mf_0', 'y': 'mf_2'}, 'mf_0')
    fis.add_rule({'x': 'mf_1', 'y': 'mf_2'}, 'mf_2')
    fis.add_rule({'x': 'mf_2', 'y': 'mf_2'}, 'mf_0')
    
    # Test inference.
    # For x nearly high (0.99) and y low (0.25), we expect y to force a low output.
    result = fis.infer({'x': 0.5, 'y': 0.99})
    print("Inference result for x=0.99, y=0.25:", result)
    
    # Plot the membership functions and the response surface.
    fis.plot_memberships(domain=(0, 1), num_points=1000)
    fis.plot_response_surface(x_range=(0, 1), y_range=(0, 1), x_points=50, y_points=50)
