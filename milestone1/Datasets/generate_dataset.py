import pandas as pd
import numpy as np
import random
from datetime import date, timedelta

# --- Configuration ---
START_DATE = date(2000, 1, 1)
END_DATE = date(2025, 10, 1)
OUTPUT_FILENAME = 'single_person_expenses_50k.csv'

# Define realistic spending categories and their typical cost ranges
CATEGORIES = {
    'Groceries': (15.0, 150.0), 'Transportation': (5.0, 75.0),
    'Eating Out': (10.0, 100.0), 'Coffee/Snacks': (3.0, 25.0),
    'Utilities': (80.0, 250.0), 'Phone/Internet': (60.0, 120.0),
    'Streaming Service': (15.99, 15.99), 'Gym Membership': (49.99, 49.99),
    'Shopping': (25.0, 500.0), 'Entertainment': (20.0, 300.0),
    'Travel': (200.0, 4000.0), 'Electronics': (100.0, 2500.0),
    'Rent/Mortgage': (1500.0, 1500.0)
}
INCOME = {'Salary': (2500.0, 2500.0)}

print("Starting dataset generation... This might take a moment.")

# --- Data Generation Logic ---
transactions = []
current_date = START_DATE
while current_date <= END_DATE:
    # Daily Expenses - INCREASED THE NUMBER OF TRANSACTIONS HERE
    for _ in range(random.randint(4, 7)): # This line was changed from (1, 4)
        category = random.choice(['Groceries', 'Transportation', 'Eating Out', 'Coffee/Snacks'])
        amount = round(random.uniform(CATEGORIES[category][0], CATEGORIES[category][1]), 2)
        transactions.append([current_date, category, 'Expense', -amount])
    # Monthly Bills
    if current_date.day == 1: transactions.append([current_date, 'Rent/Mortgage', 'Expense', -CATEGORIES['Rent/Mortgage'][0]])
    if current_date.day == 5: transactions.append([current_date, 'Utilities', 'Expense', -round(random.uniform(*CATEGORIES['Utilities']), 2)])
    if current_date.day == 15: transactions.append([current_date, 'Phone/Internet', 'Expense', -round(random.uniform(*CATEGORIES['Phone/Internet']), 2)])
    # Irregular Expenses
    if random.random() < 0.1:
        category = random.choice(['Shopping', 'Entertainment'])
        transactions.append([current_date, category, 'Expense', -round(random.uniform(*CATEGORIES[category]), 2)])
    # Bi-Weekly Income
    if current_date.day == 1 or current_date.day == 15:
        transactions.append([current_date, 'Salary', 'Income', INCOME['Salary'][0]])
    current_date += timedelta(days=1)

# --- Create DataFrame and Save ---
df = pd.DataFrame(transactions, columns=['Date', 'Category', 'Type', 'Amount'])
# Now this line will correctly trim the data to exactly 50,000 rows
df = df.sort_values(by='Date').head(50000) 
df.to_csv(OUTPUT_FILENAME, index=False)

print(f"\n✅ Success! Generated a dataset with {len(df)} rows.")
print(f"File saved as: '{OUTPUT_FILENAME}'")