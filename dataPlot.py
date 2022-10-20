import matplotlib.pyplot as py
import json

def read_json_data(path, data_type):
    file_str = open(path, 'r').read()
    json_data = json.loads(file_str)
    
    if (not (data_type == "loss" or data_type == "gradient magnitude")):
        return
    
    return json_data[data_type]

def display(data, data_type):
    X = [ a for a in range(len(data)) ]
    Y = [ float(a) for a in data ]
    
    py.plot(X, Y)
    py.xlabel("Epoch")
    py.ylabel(data_type)
    py.show()

def main():
    data_type = "loss"
    data = read_json_data("./training data/example.json", data_type)
    display(data, data_type)

if __name__ == "__main__":
    main()
    