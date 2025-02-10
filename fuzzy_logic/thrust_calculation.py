


def thrust_calculation(relative_positions_sorted, relative_heading, closure_rate, asteroids_in_distance):
        

        num_rules_az = len(az_mfs)
        num_rules_closure = len(closure_mfs)
        num_rules_size = len(size_mfs)
        num_rules_distance = len(distance_mfs)

        
        def f(x1, p1):
            """
            Computes f(x1) = -1 * (p1 * abs(x1 - 0.5) - 0.25)
            in a piecewise manner, explicitly factoring out p1*x1.
            """
            if x1 < 0.5:
                return p1 + (0.25 - 0.5 * p1)
            else:
                return -p1 + (0.25 + 0.5 * p1)

        thrust_fis_1_params = []

        for i in range(num_rules_az):
            row = []
            for j in range(num_rules_closure):
                # Example: p0, p1, and p2 are chosen based on the rule indices.
                p1 = f(relative_heading, 1)
                p2 = max(j-1,0)
                row.append([p1, p2])
            thrust_fis_1_params.append(row)
        sorted_len = len(relative_positions_sorted)

        # Thrust FIS
        thrust_sum = 0
        for i in range(min(asteroids_in_distance,sorted_len)):
            distance = relative_positions_sorted[i]
            distance_norm = math.sqrt(min(50/(distance+0.0001),0.99999))
            thrust_sum = distance_norm * ft.tsk_inference_mult(x1=relative_heading, x2=closure_rate, x1_mfs=az_mfs, x2_mfs=closure_mfs, params=thrust_fis_1_params)
            thrust_sum += thrust_sum
        # ft.plot_tsk_surface(x1_mfs=az_mfs, x2_mfs=closure_mfs, params=thrust_fis_1_params, resolution=50)
        thrust = thrust_sum * 700
        # thrust = 0
        print(thrust)