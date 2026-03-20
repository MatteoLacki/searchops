import csv
from importlib.resources import files

def open_data():
    data_path = files("searchops.data").joinpath("test_data.csv")
    with open(data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row)
