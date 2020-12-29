import pandas as pd

#############################################################################
with open('Data/Raw_data/groceries.csv') as file:
    data = [line.rstrip('\n') for line in file]
    data = [basket.split(',') for basket in data]

#############################################################################
products = []
ids = []

for id, basket_list in enumerate(data):

    # match the transaction id to the products in the basket
    ids.extend([id] * len(basket_list))

    products.extend(basket_list)
#############################################################################
df = pd.DataFrame.from_dict(

    dict(
        ids=ids,
        products=products
    )
)
#############################################################################
# each basket (transaction) should represent unique products
# this ensures that one hot encoding yields unique rows with 1 product purchased and the other values are 0 per transaction
df = df.drop_duplicates(ignore_index=True)
#############################################################################
# one hot encode the products
df = pd.get_dummies(df, columns=['products'], drop_first=False)
#############################################################################
# for each transaction id computing the sum per column (certain product),
#  allowing us to flatten the data (1 row per transaction)
df = df.groupby('ids')[df.columns.drop('ids')].sum()
#############################################################################
# patterns will be discoverd from all the transactions
df = df.drop(columns='ids')
#############################################################################
# save the processed data
df.to_parquet('Data/Processed_data/processed_basket_data.parquet',
              compression='snappy', index=False)
