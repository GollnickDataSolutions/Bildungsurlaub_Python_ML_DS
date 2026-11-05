#%% Pakete
import pandas as pd

#%% Dataframe (von Hand) erstellen
my_data = {
    'student': ['Stuart', 'Bob', 'Kevin'],
    'math': [1, 2, 3]
}
students_df = pd.DataFrame(my_data)

#%% Exportopenpyxl
students_df.to_clipboard()

#%%
students_df.to_excel("students.xlsx")
