# Model to increase the number of sequences on file to be processed by DL Models
import pandas as pd
import numpy as np


data = pd.read_csv('/Users/marcelo/Desktop/Rita/Tese/tese/Data/data-filtered-random.csv', sep=',')

input_data = np.array(data["Sequence"], data["Is transporter?"])


print(input_data)