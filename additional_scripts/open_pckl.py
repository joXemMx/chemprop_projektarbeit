import pickle
import pandas as pd

objects = []
with (open("split_indices.pckl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

train = objects[0][0]
val = objects[0][1]
test = objects[0][2]

train_df = pd.DataFrame(train)
val_df = pd.DataFrame(val)
test_df = pd.DataFrame(test)

train_df.to_csv("train_idx.csv", header=False, index=False)
val_df.to_csv("val_idx.csv", header=False, index=False)
test_df.to_csv("test_idx.csv", header=False, index=False)