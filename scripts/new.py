from eda_module import aggregate_user_data
from load_data import load_df

# Load the data
df = load_df()
df = df.head()


# # Aggregate the data
# user_aggregates = aggregate_user_data(df)
# print(user_aggregates)
print(df.columns)