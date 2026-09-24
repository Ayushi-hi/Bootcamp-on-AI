import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# --------------------------------------------------
# 1. Create gaming transactions
# --------------------------------------------------

transactions = [
    ["Sword", "Shield", "Health Potion"],
    ["Sword", "Shield"],
    ["Sword", "Shield", "Armor"],
    ["Sword", "Shield", "Health Potion"],
    ["Sword", "Armor"],
    
    ["Bow", "Arrows", "Health Potion"],
    ["Bow", "Arrows"],
    ["Bow", "Arrows", "Helmet"],
    ["Bow", "Arrows", "Health Potion"],
    ["Bow", "Arrows", "Boots"],
    
    ["Magic Staff", "Mana Potion"],
    ["Magic Staff", "Mana Potion", "Health Potion"],
    ["Magic Staff", "Mana Potion"],
    ["Magic Staff", "Mana Potion", "Armor"],
    ["Magic Staff", "Mana Potion", "Health Potion"],
    
    ["Sword", "Shield", "Health Potion", "Armor"],
    ["Sword", "Shield", "Armor"],
    ["Bow", "Arrows", "Health Potion"],
    ["Bow", "Arrows", "Boots"],
    ["Magic Staff", "Mana Potion", "Health Potion"],
    
    ["Sword", "Shield", "Helmet"],
    ["Sword", "Shield", "Health Potion"],
    ["Bow", "Arrows", "Helmet"],
    ["Bow", "Arrows", "Health Potion"],
    ["Magic Staff", "Mana Potion"],
    
    ["Sword", "Shield", "Armor"],
    ["Sword", "Shield", "Health Potion"],
    ["Bow", "Arrows", "Boots"],
    ["Magic Staff", "Mana Potion", "Health Potion"],
    ["Sword", "Shield", "Armor"]
]


# --------------------------------------------------
# 2. Display transactions
# --------------------------------------------------

print("Gaming Transactions:")
for i, transaction in enumerate(transactions, start=1):
    print(f"Player {i}: {transaction}")


# --------------------------------------------------
# 3. Convert transactions into DataFrame
# --------------------------------------------------

encoder = TransactionEncoder()

encoded_data = encoder.fit(
    transactions
).transform(transactions)

df = pd.DataFrame(
    encoded_data,
    columns=encoder.columns_
)

print("\nEncoded Dataset:")
print(df)


# --------------------------------------------------
# 4. Find frequent itemsets
# --------------------------------------------------

frequent_itemsets = apriori(
    df,
    min_support=0.20,
    use_colnames=True
)

print("\n===== FREQUENT ITEMSETS =====")
print(frequent_itemsets)


# --------------------------------------------------
# 5. Generate association rules
# --------------------------------------------------

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.60
)

print("\n===== ASSOCIATION RULES =====")

if rules.empty:
    print("No rules found. Try reducing min_threshold.")
else:

    rules = rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ]

    print(rules)


# --------------------------------------------------
# 6. Sort rules by confidence
# --------------------------------------------------

if not rules.empty:

    rules = rules.sort_values(
        by="confidence",
        ascending=False
    )

    print("\n===== RULES SORTED BY CONFIDENCE =====")
    print(rules)


# --------------------------------------------------
# 7. Simple recommendation system
# --------------------------------------------------

def recommend_item(item):

    if rules.empty:
        return

    print("\n--------------------------------")
    print("Recommendation for:", item)
    print("--------------------------------")

    found = False

    for _, rule in rules.iterrows():

        antecedents = set(rule["antecedents"])
        consequents = set(rule["consequents"])

        if item in antecedents:

            print(
                f"If player has {set(antecedents)} "
                f"→ recommend {set(consequents)}"
            )

            print(
                f"Confidence: {rule['confidence']:.2f}"
            )

            print(
                f"Lift: {rule['lift']:.2f}"
            )

            found = True

    if not found:
        print("No recommendation found.")


# --------------------------------------------------
# 8. Test recommendations
# --------------------------------------------------

recommend_item("Sword")
recommend_item("Bow")
recommend_item("Magic Staff")


# --------------------------------------------------
# 9. Visualize top rules
# --------------------------------------------------

if not rules.empty:

    top_rules = rules.head(10)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        top_rules["support"],
        top_rules["confidence"],
        s=100
    )

    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Gaming Item Association Rules")

    plt.grid(True)

    plt.show()