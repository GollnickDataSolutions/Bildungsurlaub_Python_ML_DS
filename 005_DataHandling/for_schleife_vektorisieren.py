#%% pakete
import pandas as pd
import numpy as np
import time
# %%
start_time = time.perf_counter()
max_val = 10000000
# langsam:
# [x**2 for x in range(max_val)]

# schnell, weil vektorisiert
x= np.arange(max_val)
res = x**2

stop_time = time.perf_counter()
duration = stop_time - start_time
print(f"Der Prozess hat {duration}s gedauert.")
# %%
