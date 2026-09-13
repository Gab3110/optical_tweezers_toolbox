import numpy as np
from sklearn.preprocessing import StandardScaler

#%% HELPER METHODS FOR INTERACTIVE CLASSES

def scale_columns(*arrays):
    """
    Accepts any number of 1D arrays (a, b, c, ...),
    stacks them column-wise, and returns:
        - X      (unscaled stacked array, shape (N, M))
        - X_scaled (scaled version using StandardScaler)
        - scaler (fitted StandardScaler object)

    Examples
    --------
    X, Xs, sc = scale_columns(x, y)
    X, Xs, sc = scale_columns(force, contour, extension)
    """

    # Convert all inputs to arrays
    arrays = [np.asarray(arr) for arr in arrays]

    # Verify equal length
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All input arrays must have same length. Got lengths: {lengths}"
        )

    # Stack column-wise: shape (N, M)
    X = np.column_stack(arrays)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, scaler

def option_handler(prompt, valid_values=None):
    """Helper function to get user input with optional validation."""
    while True:
        user_input = input(prompt)
        if valid_values:
            if user_input.lower() in valid_values:
                return user_input.lower()
            print(f"Invalid input. Please enter one of {valid_values}.")
        else:
            return user_input
        
def is_valid_value(prompt, number_type, min_value=None, max_value=None):
    while True:
        user_input = input(prompt)

        try:
            user_input = number_type(user_input)
        except (ValueError, TypeError):
            print("Invalid input. Please enter a valid number.")
            continue  # ask again

        if min_value is not None and user_input < min_value:
            print(f"Input should be greater than {min_value}.")
        elif max_value is not None and user_input > max_value:
            print(f"Input should be less than {max_value}.")
        else:
            return user_input
        
def is_valid_indexes(prompt, max_index):
    while True:
        user_input = input(prompt)
        try:
            indexes = [int(x) for x in user_input.split()]
            if all([ 0 <= idx <= max_index for idx in indexes]):
                return(indexes)
            else:
                print(f"Please enter values that are lower than {max_index}.")
        except ValueError:
            print("Invalid input. Please enter valid indexes separated by space.")

def is_valid_range (list_range):
    if list_range[0] > list_range[1]:
        print("Minimum must not exceed maximum.")
        return(None)
    else:
        return(list_range)

def define_fd_bounds (d, f):
    while True:
        min_distance = is_valid_value ("Minimum distance: ",
                                       float,
                                       min_value=None,
                                       max_value=None)
        max_distance = is_valid_value ("Maximum distance: ",
                                       float,
                                       min_value=None,
                                       max_value=None)
        min_force = is_valid_value ("Minimum force: ",
                                    float,
                                    min_value=None,
                                    max_value=None)
        max_force = is_valid_value ("Maximum force: ",
                                    float,
                                    min_value=None,
                                    max_value=None)
        
        # Testing ranges
        valid_dist_range = is_valid_range ([min_distance, max_distance])
        valid_force_range = is_valid_range ([min_force, max_force])
        
        if all([valid_dist_range, valid_force_range]):
            fd_bounds = [[min_distance, max_distance], [min_force, max_force]]
            return (fd_bounds)
        