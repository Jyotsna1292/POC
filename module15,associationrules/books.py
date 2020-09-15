# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:36:57 2020

@author: USER
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt

# importing dataset
books = pd.read_csv("C:/Users/USER/Downloads/book.csv")


frequent_itemsets = apriori(books, min_support=0.005, max_len=3,use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True) # 224, 2

# plot
plt.bar(list(range(1,6)),frequent_itemsets.support[1:6],color='rgmyk');plt.xticks(list(range(1,6)),frequent_itemsets.itemsets[1:6])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1) #1054,9
rules.sort_values('lift',ascending = False).head(10)


def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)

###############################################################################
# changing the min support value

frequent_itemsets = apriori(books, min_support=0.007, max_len=4,use_colnames = True) # 471, 2

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)

# plot
plt.bar(list(range(1,6)),frequent_itemsets.support[1:6],color='rgmyk');plt.xticks(list(range(1,6)),frequent_itemsets.itemsets[1:6])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1) # 4556, 9 
rules.head(20)
rules.sort_values('lift',ascending = False).head(10)


def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)










