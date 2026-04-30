import numpy as np

def load_csv(file_name = "student_performance_data.csv", delimiter = ",", skip= 1):

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


a = load_csv()
# print(a)
# print(type(a))
# print(a.shape)