import numpy as np
a = np.load("info.npy")

with open("urls.txt","w") as f:
    for item in a:
        f.write(f"{item[0]},{item[1]}\n")