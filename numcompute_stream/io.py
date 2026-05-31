import numpy as np

def load_csv(file_name = "student_performance_data.csv", delimiter = ",", skip= 1):
    """
    Load a delimited text file into a NumPy array.

    Parameters
    ----------
    file_name : str, default="student_performance_data.csv"
        Path to the input CSV or delimited text file.
    delimiter : str, default=","
        Field separator used in the file, for example ',' or '\\t'.
    skip : int, default=1
        Number of header lines to skip before reading data.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Loaded dataset represented as a NumPy array. Empty strings are
        normalised to np.nan.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file is empty after loading.

    Notes
    -----
    This function uses ``numpy.genfromtxt`` to support missing values and
    custom delimiters. The returned array is suitable for downstream
    preprocessing steps such as imputation and scaling.

    Time Complexity
    ---------------
    O(n * m), where n is the number of rows and m is the number of columns.

    Space Complexity
    ----------------
    O(n * m), required to store the loaded array in memory.
    """
    try:
        arr = np.genfromtxt(file_name,
                        delimiter= delimiter,
                        skip_header= skip,
                        filling_values= np.nan,
                        encoding = "utf-8",
                        dtype = str)
        arr[arr == ''] = np.nan
    except FileNotFoundError:
        raise FileNotFoundError("File not found")

    if arr.size == 0:
        raise ValueError("Empty csv file")
    return arr

if __name__ == "__main__":
    a = load_csv()
    print("Successfully loaded data for manual testing.")
# print(a)
# print(type(a))
# print(a.shape)
