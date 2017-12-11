import pandas as pd
import numpy as np
data = pd.read_csv("templates/data1Normed.csv")
names = list(data)
data["signcode"] = data["sign"].astype('category').cat.codes
data = data.values
Xhands = data[:, 0:63]
Xhands = Xhands.astype(np.float64)
Tsign = data[:, 64:65]
Tsign = Tsign.astype(np.int32)
for i in range(0, len(np.unique(Tsign).tolist())):
    print('{} samples in class: {}, sign: {} '.format(np.sum(Tsign==i), i, np.unique(data[:, 63]).tolist()[i]))
