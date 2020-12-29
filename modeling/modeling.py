import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
#############################################################################
basket_data = pd.read_parquet(
    'Data/Processed_data/processed_basket_data.parquet')
#############################################################################
# discover patterns from the association of products purchased
frequent_itemsets = apriori(
    basket_data,
    # rule e.g. {UHT-milk} -> {yogurth}
    # should be observed in >= 3% of all transactions
    min_support=0.03,
    use_colnames=True)
#############################################################################
# find the most meaningful association rules
rules = association_rules(
    frequent_itemsets,
    # lift(A->C) = confidence(A->C) / support(C), range: [0, inf]
    # lift > 1 indicates stronger associations
    metric="lift",
    min_threshold=1.000001)
#############################################################################
pd.set_option('display.max_columns', None)
rules.head(10)
#############################################################################
# order data so that live API requests require less computations (only filtering to find the matching antecedents)
rules = rules.sort_values(


    by=['antecedents', 'support', 'confidence', 'lift'],
    ascending=[True, False, False, False],
    ignore_index=True
)
#############################################################################
# keep only the columns needed for the API
rules = rules[['antecedents', 'consequents']]
#############################################################################
# convert values to be saved in Parquet format (frozen sets => lists)
rules = rules.assign(

    antecedents=rules['antecedents'].apply(lambda val: list(val)),

    consequents=rules['consequents'].apply(lambda val: list(val))

)
#############################################################################
# save the recommendations data
rules.to_parquet('Data/recommendations_data/recommendations.parquet',
                 compression='snappy', index=False)
