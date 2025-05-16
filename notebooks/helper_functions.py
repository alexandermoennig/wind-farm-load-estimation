import os
import warnings
import numpy as np
import pandas as pd
import numpoly
import chaospy as cp
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects 
import seaborn as sns
from sklearn.linear_model import LassoCV

pd.options.mode.chained_assignment = None
warnings.simplefilter('ignore', np.RankWarning)

def store_plot(fig, name):
    """
    Save plot figure to file with high resolution.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        name (str): Filename to save the plot as.
    """
    plot_path = os.path.dirname(os.getcwd())
    fig.savefig(os.path.join(plot_path, name), dpi=600, bbox_inches='tight')

def setup_plot(height_ratio=1.2):
    """
    Configure plot formatting and size settings.

    Args:
        height_ratio (float, optional): Figure height to width ratio. Defaults to 1.2.

    Returns:
        tuple: Figure width and height in inches.
    """
    plt.close()
    sns.set_theme(style="whitegrid", palette="bright")
    sns.set_context("paper")
    FIG_WIDTH = 16/2.54
    FIG_HEIGHT = FIG_WIDTH*height_ratio
    large_font = 9
    medium_font = 8
    small_font = 7
    x_small_font = 6
    plt.rcParams['axes.labelsize'] = medium_font 
    plt.rcParams['figure.titlesize'] = medium_font#large_font
    plt.rcParams['axes.titlesize'] = medium_font#large_font
    plt.rcParams['xtick.labelsize'] = small_font
    plt.rcParams['ytick.labelsize'] = small_font
    plt.rcParams["legend.title_fontsize"] = medium_font
    plt.rcParams['legend.fontsize'] = small_font
    plt.rcParams['font.size'] = x_small_font
    plt.rcParams['font.family'] = 'Georgia'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Georgia'
    plt.rcParams['mathtext.it'] = 'Georgia:italic'
    plt.rcParams['mathtext.bf'] = 'Georgia:bold'
    return FIG_WIDTH, FIG_HEIGHT

def print_dimensions(g):
    """
    Print FacetGrid figure dimensions in centimeters.

    Args:
        g (sns.FacetGrid): Seaborn FacetGrid object.
    """
    fig_in = g.fig.get_size_inches()
    # Convert the dimensions from inches to centimeters (1 inch = 2.54 cm)
    fig_cm = fig_in * 2.54
    print(f"FacetGrid dimensions: {fig_cm[0]:.2f} cm wide by {fig_cm[1]:.2f} cm tall")

def customize_ticks(g, axis='x', rotate=False, nth_label=1):
    """
    Customizes tick labels on the x or y-axis by showing only every nth label.

    Parameters:
    - g: sns.FacetGrid object
    - axis: 'x' for x-axis, 'y' for y-axis
    - rotate: True to rotate labels, False to leave them horizontal
    - nth_label: an integer to show every nth label
    """
    for ax in g.axes.flatten():
        if axis == 'x':
            labels = ax.get_xticklabels()
            new_labels = [label if (i % nth_label == 0) else '' for i, label in enumerate(labels)]
            ax.set_xticklabels(new_labels, rotation=45 if rotate else 0)
        elif axis == 'y':
            labels = ax.get_yticklabels()
            new_labels = [label if (i % nth_label == 0) else '' for i, label in enumerate(labels)]
            ax.set_yticklabels(new_labels, rotation=45 if rotate else 0)

def optimize_facetgrid_titles(facet_grid, row_pre="", col_pre="", row_post=""):
    """
    Optimizes the titles of a Seaborn FacetGrid.

    Args:
        facet_grid (sns.FacetGrid): The FacetGrid object to be optimized.
        row_pre (str, optional): Prefix for row titles. Defaults to "".
        col_pre (str, optional): Prefix for column titles. Defaults to "".
        row_post (str, optional): Suffix for row titles. Defaults to "".

    Returns:
        sns.FacetGrid: The optimized FacetGrid object.
    """
    
    # Remove the default texts
    [plt.setp(ax.texts, text="") for ax in facet_grid.axes.flat]
    
    # Construct row title template based on row_pre
    row_title_template = f"{row_pre}: " * bool(row_pre) + "{row_name}" + f" {row_post}" * bool(row_post)
    col_title_template = f'{col_pre}: ' + '{col_name}' if col_pre else '{col_name}'
    # Set custom titles for rows and columns
    facet_grid.set_titles(row_template=row_title_template, col_template=col_title_template)
    
    return facet_grid

def add_median_labels(ax, fmt=".1f"):
    """
    Add value labels to median lines in a box plot.

    Args:
        ax (matplotlib.axes.Axes): Axes object containing the box plot.
        fmt (str, optional): Format string for median values. Defaults to ".1f".
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4: len(lines): lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] -
                    median.get_xdata()[0]) == 0 else y
        text = ax.text(
            x,
            y,
            f"{value:{fmt}}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )
        # create median-colored border around white text for contrast
        text.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ]
        )
        
def filter_data_by_year(data, years):
    """
    Filter DataFrame to include only specified years.

    Args:
        data (DataFrame): Input data with 'Zeitstempel' datetime column.
        years (list): Years to keep in the filtered data.

    Returns:
        DataFrame: Filtered data containing only the specified years.
    """
    if data is None:
        return pd.DataFrame()

    return data[data["Zeitstempel"].dt.year.isin(years)]

def load_turbine_data(turbine_number, path):
    """
    Load turbine operational data from feather file.

    Args:
        turbine_number (int): Turbine ID.
        path (str): Directory path containing the feather files.
        
    Returns:
        DataFrame: Operational data for specified turbine.
    """
    return pd.read_feather(
        os.path.join(path, f"Turb 0{turbine_number}_op_data.feather")
    )

def load_and_filter_turbine_data(turbine_number, years, path, datatype="10m"):
    """
    Load and filter turbine data by years and data type.

    Args:
        turbine_number (int): Turbine ID.
        years (list): Years to keep in filtered data.
        path (str): Base directory path.
        datatype (str, optional): Type of data to load ('10m' or 'status'). Defaults to '10m'.

    Returns:
        DataFrame: Filtered turbine data. Empty DataFrame if no data found.
    """

    if datatype == "10m":
        df = pd.read_feather(
            os.path.join(path, datatype, f"Turb 0{turbine_number}_op_data.feather")
        )
        df = df[df["Zeitstempel"].dt.year.isin(years)].copy()
    elif datatype == "status":
        df = pd.read_feather(
            os.path.join(path, datatype, f"Turb 0{turbine_number}_alarm_data.feather")
        )
        df = df[df["Start"].dt.year.isin(years)].copy()

    if df is None:
        df = pd.DataFrame()
        
    return df

def custom_apply(row, df_check):
    """
    Check if alarm code/subcode pair exists in reference dataframe.

    Args:
        row (Series): Row containing 'Code' and 'Subcode'.
        df_check (DataFrame): Reference data with 'Main Code' and 'Sub Code' columns.

    Returns:
        bool: True if code pair exists in reference data, False otherwise.
    """
    if row["Code"] in df_check["Main Code"].values:
        if pd.notna(row["Code"]) and (row["Code"], row["Subcode"]) in zip(
            df_check["Main Code"], df_check["Sub Code"]
        ):
            return True
        elif not pd.notna(row["Subcode"]):
            return True
        else:
            return False
    else:
        return False

def prefilter(operation_df, alarm_df, filter_by="loglist", option="onlyAlarm", loglist_file=None):
    """
    Filter operational data based on alarm logs using specified criteria.

    Args:
        operation_df (DataFrame): Operational data of turbine.
        alarm_df (DataFrame): Alarm logs of turbine.
        filter_by (str, optional): Filter method ('loglist' or 'Prio'). Defaults to 'loglist'.
        option (str, optional): Filter option ('onlyAlarm' or 'Priority'). Defaults to 'onlyAlarm'.
        loglist_file (str, optional): Path to loglist Excel file. Required if filter_by='loglist'.

    Returns:
        DataFrame: Filtered operational data with added 'Category' column.
    """
    
    alarm_df = alarm_df[
        [
            "Code",
            "Subcode",
            "Beschreibung",
            "Start",
            "Ende",
            "Dauer",
            "Prio.",
            "Verf.-Kat.",
        ]
    ]

    if filter_by == "loglist":
        if loglist_file is None:
            raise Exception('No loglist file provided!')
        df_check = pd.read_excel(
            loglist_file
        )

        if option == "onlyAlarm":
            df_check = df_check.loc[df_check["Type"] == "Alarm"]

        alarm_df["Critical"] = alarm_df.apply(
            lambda row: custom_apply(row, df_check), axis=1
        )

        alarm_df = alarm_df.loc[alarm_df["Critical"]]
        alarm_df = alarm_df.iloc[:, 0:-2]
    elif filter_by == "Prio":
        alarm_df = alarm_df.loc[alarm_df["Prio."] >= int(option)]

    start_date = operation_df["Referenzzeitstempel"].min()
    end_date = operation_df["Referenzzeitstempel"].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq="10T")
    control_series = pd.Series(date_range)
    control_array = np.array(control_series)

    start_times = pd.to_datetime(alarm_df["Start"])
    end_times = pd.to_datetime(alarm_df["Ende"])

    start_times = start_times.values[:, np.newaxis]
    end_times = end_times.values[:, np.newaxis]

    overlap = ((start_times <= control_array) & (end_times >= control_array)) | (
        (start_times >= control_array)
        & (start_times < control_array + np.timedelta64(10, "m"))
    )

    control_result = np.any(overlap, axis=0)
    control_result = pd.Series(control_result, index=control_series)
    flagged_share = control_result.value_counts(normalize=True)
    print(
        f"{flagged_share.loc[flagged_share.index==True].item()*100:.2f} % have been flagged as abnormal"
    )
    flag = control_result[operation_df["Zeitstempel"]]
    operation_df.set_index("Zeitstempel", inplace=True)
    operation_df["Category"] = flag
    return operation_df

def derivative(f, a, method="central", h=0.01):
    """
    Compute the difference formula for f'(a) with step size h.

    Args:
        f (function): Vectorized function of one variable.
        a (float): Compute derivative at x = a.
        method (str, optional): Difference formula ('forward', 'backward' or 'central'). 
                                Defaults to 'central'.
        h (float, optional): Step size in difference formula. Defaults to 0.01.

    Returns:
        float: Difference formula value:
            - central: f(a+h) - f(a-h))/2h
            - forward: f(a+h) - f(a))/h
            - backward: f(a) - f(a-h))/h
    """

    if method == "central":
        return (f(a + h) - f(a - h)) / (2 * h)
    elif method == "forward":
        return (f(a + h) - f(a)) / h
    elif method == "backward":
        return (f(a) - f(a - h)) / h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")

def PetersonMethod_SD(SD_p, wsp, power_curve, B=0.7):
    """
    Convert power standard deviation to wind standard deviation using Peterson Method.

    Args:
        SD_p (float): Power standard deviation.
        wsp (float): Wind speed value.
        power_curve (function): Power curve function.
        B (float, optional): Peterson factor. Defaults to 0.7.

    Returns:
        float: Wind speed standard deviation.
    """
    # Calculate the power curve values at the wind speeds
    # power_curve_values = power_curve(wsp)

    # Calculate the central differences for the derivative of the power curve
    # d_power_curve = np.gradient(power_curve_values, wsp, edge_order=2)
    d_power_curve = derivative(power_curve, wsp)
    # Add a small epsilon value to avoid division by zero
    epsilon = 1e-8
    d_power_curve[d_power_curve == 0] = epsilon

    # Calculate the wind standard deviation
    sd_u = SD_p / (B * d_power_curve)

    return sd_u

def get_air_density(T, h):
    """
    Calculate air density based on temperature and elevation (Manwell et al., 2009).

    Args:
        T (float): Temperature in deg Celsius.
        h (float): Elevation in m.

    Returns:
        float: Air density in kg/m3, rounded to 4 decimal places.
    """
    # Get the pressure in kPa
    T = T + 273.15  # Convert to Kelvin
    p = 101.29 - (0.011837) * h + (4.793e-7) * h**2
    rho = 3.4837 * (p/T)
    return round(rho, 4)

def get_envelope(data, signal, order_upper=3, order_lower=3, multiplier=1):
    """
    Calculate upper and lower polynomial envelopes for a signal based on binned wind speed.

    Args:
        data (DataFrame): Input data with wind speed bins.
        signal (str): Name of signal column to analyze.
        order_upper (int, optional): Order of upper envelope polynomial. Defaults to 3.
        order_lower (int, optional): Order of lower envelope polynomial. Defaults to 3.
        multiplier (float, optional): Standard deviation multiplier. Defaults to 1.

    Returns:
        tuple: Lower and upper envelope polynomial functions.
    """
    env_df = pd.DataFrame(data.groupby('wsp_binned')[signal].max(
    ) + multiplier*data.groupby('wsp_binned')[signal].std())
    env_df_min = data.groupby('wsp_binned')[signal].min(
    ) - multiplier*data.groupby('wsp_binned')[signal].std()
    env_df.rename(columns={signal: "Max"}, inplace=True)
    env_df["Min"] = env_df_min
    binwidth = env_df.index.values[1] - env_df.index.values[0]
    env_df["X"] = env_df.index.values.astype(float) + binwidth/2
    env_df.reset_index(inplace=True, drop=True)
    env_df = env_df.sort_values(by="X")
    env_df.dropna(how="any", axis=0, inplace=True)

    lower_coeffs = np.polyfit(env_df['X'], env_df['Min'], order_lower)
    upper_coeffs = np.polyfit(env_df['X'], env_df['Max'], order_upper)
    lower_env = np.poly1d(lower_coeffs)
    upper_env = np.poly1d(upper_coeffs)
    return lower_env, upper_env

def eval_polynomial(x_val, poly_coeffs, order=3):
    """
    Evaluate polynomial at given x values.

    Args:
        x_val (float or array): X values to evaluate polynomial at.
        poly_coeffs (array): Polynomial coefficients.
        order (int, optional): Order of polynomial. Defaults to 3.

    Returns:
        float or array: Polynomial values at x_val.
    """
    result = 0
    for i, coeff in enumerate(poly_coeffs):
        result += coeff * (x_val ** (order - i))
    return result

def fit_regression_lasso_cv(
    polynomials,
    abscissas,
    evals,
    cv_fit=10,
    retall=0,
    seed=np.random.randint(0, 1000)):
    """
    Fit a LASSO regression model with cross-validation. Adapted from chaospy repository.

    Args:
        polynomials (numpoly.ndpoly): Polynomial basis functions.
        abscissas (array): Sample points to evaluate polynomials at.
        evals (array): Target values to fit.
        cv_fit (int, optional): Number of cross-validation folds. Defaults to 10.
        retall (int, optional): Return option (0-3):
            - 0: Returns only approximation model
            - 1: Returns (model, coefficients)
            - 2: Returns (model, coefficients, polynomial evaluations)
            - 3: Returns (model, alpha_cv_data, final_alpha)
        seed (int, optional): Random seed for cross-validation. Defaults to random int.

    Returns:
        Various: Depending on retall value:
    """

    abscissas = np.atleast_2d(abscissas)
    assert abscissas.ndim == 2, "too many dimensions"

    polynomials = numpoly.aspolynomial(polynomials)

    evals = np.asarray(evals)
    assert abscissas.shape[-1] == len(evals)

    poly_evals = polynomials(*abscissas).T
    shape = evals.shape[1:]
    if shape:
        evals = evals.reshape(len(evals), -1)

    model = LassoCV(cv=cv_fit, n_alphas=100, eps=1e-10,
                    random_state=seed, n_jobs=-1, fit_intercept=False)
    model.fit(poly_evals, evals)
    uhat = np.transpose(model.coef_)

    # Create DataFrame to store alpha and criterion values
    alpha_cv_df = {
        'alpha_values': model.alphas_,
        'mse_path': model.mse_path_,
    }
    final_alpha = model.alpha_
    approx_model = numpoly.sum((polynomials * uhat.T), -1).reshape(shape)

    choices = {
        0: approx_model,
        1: (approx_model, uhat),
        2: (approx_model, uhat, poly_evals),
        3: (approx_model, alpha_cv_df, final_alpha),  # Added DataFrame
    }

    return choices[retall]

def get_joint(input_path_scada):
    """
    Generate a joint distribution of wind conditions, turbulence intensity,
    air density, and yaw misalignment based on SCADA data.

    Args:
        input_path_scada (str): Path to directory containing SCADA data feather file.

    Returns:
        cp.J: Joint distribution containing:
            - Beta distribution for wind speed (3-25 m/s)
            - Uniform distribution for turbulence intensity
            - Uniform distribution for air density
            - Uniform distribution for yaw misalignment
    """

    # Constants specific to the SCADA dataset and distribution setup
    INPUT_PATH_SCADA = input_path_scada
    FILENAME = 'farmdata_2017-2022_refined.feather'
    ALPHA = 1.02
    BETA = 3
    MIN_WIND_SPEED = 3
    MAX_WIND_SPEED = 25
    TI_ORDER_UPPER = 3
    TI_ORDER_LOWER = 2
    YAW_ORDER = 18

    # Read the dataset
    df = pd.read_feather(os.path.join(INPUT_PATH_SCADA, FILENAME))

    # Preprocessing and calculations
    df = df.loc[~df['is_Abnormal']]

    lower_ti, upper_ti = get_envelope(
        df, 'TI_est', order_upper=TI_ORDER_UPPER, order_lower=TI_ORDER_LOWER)
    lower_rho, upper_rho = get_envelope(df, 'Air Density [kg/m3]')
    lower_yaw, upper_yaw = get_envelope(df, 'Relative wind direction (wind shear) (avg.) [°]',
                                        order_upper=YAW_ORDER, order_lower=YAW_ORDER)

    # Create distributions for each variable
    dist_wind = cp.Beta(ALPHA, BETA, MIN_WIND_SPEED, MAX_WIND_SPEED)
    dist_ti = cp.Uniform(eval_polynomial(dist_wind, lower_ti, TI_ORDER_LOWER),
                        eval_polynomial(dist_wind, upper_ti, TI_ORDER_UPPER))
    dist_rho = cp.Uniform(eval_polynomial(dist_wind, lower_rho),
                        eval_polynomial(dist_wind, upper_rho))
    dist_yaw = cp.Uniform(eval_polynomial(dist_wind, lower_yaw, YAW_ORDER),
                        eval_polynomial(dist_wind, upper_yaw, YAW_ORDER))

    # Combine the distributions into a joint distribution
    joint = cp.J(dist_wind, dist_ti, dist_rho, dist_yaw)
    return joint