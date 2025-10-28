import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
# Categorize
CATEGORIES_KEYWORDS = {
    'Housing': ['rent', 'mortgage', 'property tax', 'home insurance', 'hoa', 'plumbing', 'electrician', 'repair', 'furniture', 'ikea', 'home depot'],
    'Transportation': ['uber', 'car', 'lyft', 'taxi', 'bus', 'subway', 'amtrak', 'train', 'fuel', 'gasoline', 'car payment', 'car insurance', 'parking', 'car wash', 'auto repair', 'dmv', 'bolt'],
    'Groceries & Household': ['grocery', 'groceries', 'market', 'safeway', 'kroger', 'walmart', 'costco', 'sprouts', 'trader joe', 'publix', 'food', 'supermarket', 'target', 'whole foods', 'household supplies', 'toilet paper', 'soap', 'detergent'],
    'Dining': ['restaurant', 'cafe', 'coffee', 'snaks', 'starbucks', 'doordash', 'grubhub', 'ubereats', 'food delivery', 'mcdonalds', 'burger king', 'pizza hut', 'dominos', 'chipotle', 'eats', 'eating out'],
    'Entertainment': ['movie', 'cinema', 'concert', 'spotify', 'netflix', 'hulu', 'disney+', 'app store', 'google play', 'tickets', 'bar', 'nightclub', 'apple music', 'youtube premium', 'gaming', 'steam', 'playstation', 'xbox'],
    'Personal & Family Care': ['haircut', 'shopping' ,'salon', 'barber', 'cosmetics', 'toiletries', 'sephora', 'ulta', 'gym', 'fitness', 'yoga', 'childcare', 'daycare', 'baby', 'pet food', 'vet', 'veterinarian', 'spa', 'massage'],
    'Work & Education': ['office supplies', 'stationery', 'udemy', 'coursera', 'book', 'textbook', 'tuition', 'school', 'college', 'webinar', 'software', 'adobe', 'slack', 'zoom', 'linkedin learning', 'github'],
    'Health & Medical': ['doctor', 'dentist', 'hospital', 'pharmacy', 'cvs', 'walgreens', 'rite aid', 'medicine', 'prescription', 'health insurance', 'copay', 'vision', 'therapy', 'physician'],
    'Travel': ['flight', 'airline', 'american airlines', 'delta', 'united', 'southwest', 'hotel', 'airbnb', 'booking.com', 'expedia', 'vacation', 'trip', 'luggage', 'rental car', 'hertz', 'avis'],
    'Technology & Communication': ['phone bill', 'verizon', 'at&t', 't-mobile', 'internet', 'comcast', 'xfinity', 'google fi', 'gadget', 'apple', 'samsung', 'best buy', 'newegg', 'aws', 'gcp', 'azure', 'google domain'],
    'Financial & Insurance': ['life insurance', 'bank fee', 'atm fee', 'financial advisor', 'investment', 'stock', 'coinbase', 'robinhood', 'loan payment', 'student loan', 'credit card payment', 'transfer'],
    'Business Expenses': ['client dinner', 'business travel', 'consulting', 'legal fee', 'advertising', 'quickbooks', 'adwords'],
    'Taxes': ['tax return', 'irs', 'property tax', 'income tax', 'tax prep', 'h&r block', 'turbotax'],
    'Income': ['salary', 'paycheck', 'deposit', 'bonus', 'freelance', 'invoice', 'refund', 'reimbursement', 'interest income', 'dividend'],
    'Other': ['charity', 'donation', 'gift',]
}

# Manual logs
def categorize_transaction(description, trans_type):
    if not isinstance(description, str): return 'Other'
    description_lower = description.lower()
    if trans_type == "Income":
        if any(keyword in description_lower for keyword in CATEGORIES_KEYWORDS['Income']): return 'Income'
        return 'Income'
    for category, keywords in CATEGORIES_KEYWORDS.items():
        if category == 'Income': continue
        if any(keyword in description_lower for keyword in keywords): return category
    return 'Other'

# Upload CSV
def process_uploaded_data(df_raw, username):
    df = df_raw.copy()

    date_col_original = None
    if 'Date' in df.columns: date_col_original = 'Date'
    elif 'date' in df.columns: date_col_original = 'date'

    if date_col_original:
        initial_rows = len(df)
        df.dropna(subset=[date_col_original], inplace=True)
        if pd.api.types.is_string_dtype(df[date_col_original]):
             df = df[df[date_col_original].str.strip() != '']

        rows_dropped = initial_rows - len(df)
        print(f"DEBUG: Dropped {rows_dropped} rows with missing/blank initial '{date_col_original}' value.")
        if df.empty:
            st.error("Upload failed: No rows with valid date entries found after initial cleanup.")
            return None
    else:
         st.error("Upload failed: Could not find 'Date' or 'date' column.")
         return None

    original_category_col_name = None
    if 'Category' in df.columns:
        original_category_col_name = 'category_original'
        df.rename(columns={'Category': original_category_col_name}, inplace=True)
    elif 'category' in df.columns:
        original_category_col_name = 'category_original'
        df.rename(columns={'category': original_category_col_name}, inplace=True)

    df.columns = [str(col).lower().strip() for col in df.columns] #lowercase

    required_cols = ['date', 'amount']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Upload failed. Required columns 'date', 'amount' not found. Found: {list(df.columns)}")
        return None

    # date Conversion
    try:
        df['date_converted'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        nat_count = df['date_converted'].isnull().sum()
        print(f"DEBUG: Conversion attempted on {len(df)} rows. Found {nat_count} invalid dates (NaT).")
        st.session_state.invalid_date_count = nat_count

        if nat_count > 0:
             st.warning(f"{nat_count} dates had the wrong format (expected DD-MM-YYYY) and were removed.")

        df['date'] = df['date_converted']
        df.dropna(subset=['date'], inplace=True)
        if df.empty:
            st.warning("All rows were removed due to invalid date formats during conversion.")
            return None

    except Exception as e:
        st.error(f"Error during date conversion: {e}")
        return None

    df['Type'] = 'Expense'
    if 'type' in df.columns:
        df.loc[df['type'].str.contains('income', case=False, na=False), 'Type'] = 'Income'
    elif 'type' not in df.columns:
         df.loc[df['amount'] > 0, 'Type'] = 'Income'
    df['amount'] = df['amount'].abs()

    if 'description' not in df.columns: df['description'] = ""
    df['description'] = df['description'].fillna("").astype(str)
    originally_blank_description = df['description'].str.strip() == ""
    df['category'] = 'Other'

    for category, keywords in CATEGORIES_KEYWORDS.items():
        pattern = '|'.join(filter(None, keywords))
        if not pattern: continue
        try:
            mask = ~originally_blank_description & df['description'].str.contains(pattern, case=False, na=False, regex=True)
            if category == 'Income':
                df.loc[mask & (df['Type'] == 'Income') & (df['category'] == 'Other'), 'category'] = 'Income'
            else:
                df.loc[mask & (df['Type'] == 'Expense') & (df['category'] == 'Other'), 'category'] = category
        except Exception as e:
            st.warning(f"Regex error for '{category}': {e}")
            continue

    # get back its original category if description was blank
    if original_category_col_name is not None and original_category_col_name in df.columns:
         valid_original = df[original_category_col_name].notna() & (df[original_category_col_name].astype(str).str.strip() != '')
         df.loc[originally_blank_description & valid_original, 'category'] = df[original_category_col_name]
         df['category'] = df['category'].astype(str).str.strip()

    # Filling blank descriptions based on final category
    blank_desc_mask = df['description'].str.strip() == ""
    valid_category_mask = (df['category'] != 'Other') & (df['category'] != '') & (df['category'].notna())
    df.loc[blank_desc_mask & valid_category_mask, 'description'] = "Uploaded: " + df['category']
    df.loc[df['description'].str.strip() == "", 'description'] = "Uploaded Data" # Generic fallback

    # Final column preps
    df.rename(columns={'date': 'Date', 'amount': 'Amount', 'description': 'Description', 'category': 'Category'}, inplace=True)
    df['id'] = [f"csv_{time.time()}_{i}" for i in range(len(df))]
    df['User'] = username
    final_cols = ['id', 'Date', 'Type', 'Amount', 'Description', 'Category', 'User']
    df_final = df[[col for col in final_cols if col in df.columns]].copy()
    print("--- End process_uploaded_data ---\n")
    return df_final.to_dict('records')


if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'transactions' not in st.session_state: st.session_state.transactions = []
if 'uploaded_file_processed' not in st.session_state: st.session_state.uploaded_file_processed = False
if 'current_uploaded_file' not in st.session_state: st.session_state.current_uploaded_file = None
if 'invalid_date_count' not in st.session_state: st.session_state.invalid_date_count = 0

st.set_page_config(page_title="BudgetWise", page_icon="", layout="wide")
st.markdown("""<style>.stApp { background: linear-gradient(to right top, #d3e0ff, #e4eaff, #f5f5ff, #ffffff, #ffffff); } h1, h2, h3 { color: #0d47a1; } .st-emotion-cache-91n1wr p { font-weight: bold; color: #1565c0; } </style>""", unsafe_allow_html=True)

# login
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; color: #0d47a1;'>Welcome to BudgetWise</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("Please log in")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if password == "123" and username: # Demo login
                st.session_state.logged_in = True; st.session_state.username = username
                st.success("Logged in!"); time.sleep(1); st.rerun()
            else: st.error("Invalid credentials")

# Display
else:
    st.sidebar.title(f"Welcome, {st.session_state.username}!")
    def handle_logout():
        keys_to_keep = []
        for key in list(st.session_state.keys()):
             if key not in keys_to_keep: del st.session_state[key]
        st.rerun()
    st.sidebar.button("Logout", on_click=handle_logout)
    st.sidebar.markdown("---"); 

    
    st.title("Your Budgetwise")

    with st.expander("Upload Expense History", expanded=True):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key='file_uploader_widget')

        
        if uploaded_file is not None:
            if uploaded_file != st.session_state.current_uploaded_file or not st.session_state.uploaded_file_processed:
                st.session_state.current_uploaded_file = uploaded_file
                st.session_state.invalid_date_count = 0 
                try:
                    with st.spinner('Reading and processing...'):
                        df_read = pd.read_csv(uploaded_file, low_memory=False) # Use low_memory=False
                        #print(f"DEBUG TERMINAL: pd.read_csv initial shape: {df_read.shape}") 

                        new_data = process_uploaded_data(df_read, st.session_state.username)

                    if new_data is not None:
                        st.session_state.transactions = new_data
                        st.session_state.uploaded_file_processed = True
                        st.success(f"Successfully processed {len(new_data)} transactions!")
                        if st.session_state.invalid_date_count > 0:
                            st.warning(f"{st.session_state.invalid_date_count} rows were removed due to invalid date formats (expected DD-MM-YYYY).")
                        st.rerun() # Rerun 
                    else:
                        st.session_state.current_uploaded_file = None
                        st.session_state.uploaded_file_processed = False
                        st.session_state.transactions = []

                except Exception as e:
                    st.error(f"Error during file upload/processing: {e}")
                    st.exception(e)
                    st.session_state.current_uploaded_file = None
                    st.session_state.uploaded_file_processed = False
                    st.session_state.transactions = []

        elif st.session_state.current_uploaded_file is not None:
             st.info("File removed. Clearing data.")
             st.session_state.current_uploaded_file = None
             st.session_state.uploaded_file_processed = False
             st.session_state.transactions = []
             st.session_state.invalid_date_count = 0
             st.rerun()


    # Manual logs
    st.header("Log your transactions")
    with st.form(key="transaction_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1: trans_date = st.date_input("Date", date.today())
        with col2: amount = st.number_input("Amount ($)", min_value=0.01, step=0.01)
        with col3: trans_type = st.selectbox("Type", ["Expense", "Income"])
        description = st.text_input("Description")
        submit_button = st.form_submit_button(label="Add Transaction", use_container_width=True)

    if submit_button:
        if not description: st.error("Please enter a description.")
        else:
            category = categorize_transaction(description, trans_type)
            new_transaction = { "id": str(time.time()), "Date": trans_date, "Type": trans_type, "Amount": amount, "Description": description, "Category": category, "User": st.session_state.username }
            st.session_state.transactions.append(new_transaction)
            st.success(f"Added {trans_type}: ${amount}")
            st.rerun()

    st.markdown("---")

    st.header("Your Financial Analysis")

    # Display main dashboard content only if transactions exist
    if st.session_state.transactions:
        with st.spinner("Analyzing data & building charts..."):
            time.sleep(0.1)

            if not isinstance(st.session_state.transactions, list) or not all(isinstance(item, dict) for item in st.session_state.transactions):
                st.error("Internal data error."); st.stop()
            df = pd.DataFrame(st.session_state.transactions)
            if df.empty: st.info("No transactions to analyze."); st.stop()

            try:
                if 'Date' not in df.columns: raise ValueError("'Date' column missing")
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                null_dates_final = df['Date'].isnull().sum()
                if null_dates_final > 0:
                     st.warning(f"Note: {null_dates_final} transactions have invalid dates and are excluded from analysis.")
                     df.dropna(subset=['Date'], inplace=True)
                if df.empty: raise ValueError("No valid dates remaining after final check.")
            except Exception as e:
                st.error(f"Date error before analysis: {e}"); st.stop()

            expenses_df = df[df['Type'] == 'Expense'].copy()

            
            st.subheader("Income vs. Expense")
            total_income = df[df['Type'] == 'Income']['Amount'].sum()
            total_expense = expenses_df['Amount'].sum()
            net_savings = total_income - total_expense
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Income", f"${total_income:,.2f}")
            col2.metric("Total Expense", f"${total_expense:,.2f}")
            col3.metric("Net Savings", f"${net_savings:,.2f}", delta=f"{net_savings:,.2f}", delta_color="normal" if net_savings >= 0 else "inverse")

            # --- B. Charts ---
            if not expenses_df.empty:
                st.subheader("Expense Analysis")
                col1, col2 = st.columns(2)
                # Pie Chart
                with col1:
                    st.write("Spending by Category (Pie)")
                    category_spending = expenses_df.groupby('Category')['Amount'].sum()
                    if not category_spending.empty:
                        value_threshold = category_spending.sum() * 0.02
                        small_categories_sum = category_spending[category_spending < value_threshold].sum()
                        main_categories = category_spending[category_spending >= value_threshold]
                        if small_categories_sum > 0 and not main_categories.empty: main_categories['Other (<2%)'] = small_categories_sum
                        elif small_categories_sum > 0: main_categories = pd.Series({'Other (<2%)': small_categories_sum})
                        if not main_categories.empty:
                            fig, ax = plt.subplots(figsize=(10, 8)); palette = sns.color_palette('pastel', len(main_categories))
                            try: ax.pie(main_categories, labels=main_categories.index, autopct='%1.1f%%', startangle=90, colors=palette); ax.axis('equal'); st.pyplot(fig)
                            except Exception as e: st.error(f"Pie chart error: {e}")
                        else: st.info("No data for pie chart.")
                    else: st.info("No category spending.")
                # Bar Chart
                with col2:
                    st.write("Spending Summary (Bar)")
                    category_spending_df = expenses_df.groupby('Category')['Amount'].sum().reset_index(name="Total Amount")
                    if not category_spending_df.empty: st.bar_chart(category_spending_df, x="Category", y="Total Amount")
                    else: st.info("No category spending.")

                # Monthly Line Chart
                st.subheader("Total Monthly Spending (All Time)")
                try:
                    expenses_dated = expenses_df.set_index('Date')
                    if isinstance(expenses_dated.index, pd.DatetimeIndex):
                        expenses_by_month = expenses_dated.resample('ME')['Amount'].sum()
                        if not expenses_by_month.empty: st.line_chart(expenses_by_month)
                        else: st.info("No monthly spending.")
                    else: st.warning("Date index issue for monthly chart.")
                except Exception as e: st.error(f"Monthly chart error: {e}")

                # --- C. Yearly Table ---
                st.subheader("Monthly Breakdown by Category (By Year)")
                try:
                    if pd.api.types.is_datetime64_any_dtype(expenses_df['Date']):
                        expenses_df['Year'] = expenses_df['Date'].dt.year
                        all_years = sorted(expenses_df['Year'].unique(), reverse=True)
                        if all_years:
                            selected_year = st.selectbox("Select Year:", all_years)
                            yearly_expenses = expenses_df[expenses_df['Year'] == selected_year]
                            if not yearly_expenses.empty:
                                yearly_expenses = yearly_expenses.copy(); yearly_expenses['Month'] = yearly_expenses['Date'].dt.strftime('%Y-%m')
                                monthly_summary = yearly_expenses.pivot_table(index='Category', columns='Month', values='Amount', aggfunc='sum', fill_value=0)
                                st.dataframe(monthly_summary)
                            else: st.info(f"No data for {selected_year}.")
                        else: st.info("No yearly data.")
                    else: st.warning("Cannot extract year.")
                except Exception as e: st.error(f"Yearly breakdown error: {e}")
            else:
                st.info("Add expenses to see analysis.")
        # End spinner

        st.markdown("---") # Separator

        # --- Interactive Transaction Editor Section ---
        st.header("Manage Transactions")
        try:
            # Prepare DataFrame for editor just before displaying it
            if isinstance(st.session_state.transactions, list) and all(isinstance(item, dict) for item in st.session_state.transactions):
                 all_data_df = pd.DataFrame(st.session_state.transactions)
                 if 'id' not in all_data_df.columns: all_data_df['id'] = [str(time.time() + i) for i in range(len(all_data_df))]
                 all_data_df['id'] = all_data_df['id'].astype(str)
                 if 'Date' in all_data_df.columns:
                     all_data_df['Date'] = pd.to_datetime(all_data_df['Date'], errors='coerce')
                     all_data_df.dropna(subset=['Date'], inplace=True) # Ensure editor gets valid dates
                     all_data_df = all_data_df.sort_values(by="Date", ascending=False, na_position='last').reset_index(drop=True)
                 else:
                      st.warning("Sorting disabled: 'Date' missing."); all_data_df = all_data_df.reset_index(drop=True)

                 ROWS_PER_PAGE = 50
                 total_rows = len(all_data_df)
                 is_large_dataset = total_rows > ROWS_PER_PAGE * 10 # Define large dataset threshold

                 if total_rows == 0: st.info("No transactions to edit."); df_for_editor = pd.DataFrame() # Handle empty case
                 else:
                     # Apply pagination if dataset is large
                     if is_large_dataset:
                         # st.info(f"{total_rows} txns. Paginated ({ROWS_PER_PAGE}/page). Add/delete disabled.")
                         total_pages = max(1, (total_rows // ROWS_PER_PAGE) + (1 if total_rows % ROWS_PER_PAGE > 0 else 0))
                         page_number = st.slider("Page", 1, total_pages, 1)
                         start_idx = (page_number - 1) * ROWS_PER_PAGE; end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
                         start_idx = max(0, start_idx); end_idx = min(total_rows, end_idx)
                         df_for_editor = all_data_df.iloc[start_idx:end_idx] if start_idx < end_idx else pd.DataFrame(columns=all_data_df.columns)
                     else: # Show all data if not large
                         st.info("Double-click to edit. Select rows & press Delete.")
                         df_for_editor = all_data_df

                     all_categories = list(CATEGORIES_KEYWORDS.keys()) # Get category list for dropdown

                     # Display the editor if there's data for the current view
                     if isinstance(df_for_editor, pd.DataFrame) and not df_for_editor.empty:
                        edited_df = st.data_editor(
                            df_for_editor, key="data_editor", num_rows="dynamic" if not is_large_dataset else "fixed", disabled=["User", "id"],
                            column_config={ # Define how each column should behave in the editor
                                "id": None, "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                                "Amount": st.column_config.NumberColumn("Amount", format="$%.2f", required=True),
                                "Category": st.column_config.SelectboxColumn("Category", options=all_categories, required=True),
                                "Type": st.column_config.SelectboxColumn("Type", options=["Expense", "Income"], required=True),
                                "Description": st.column_config.TextColumn("Description", required=False) # Description is optional
                             }, width='stretch', hide_index=True,
                        )

                        # --- Logic to Save Edits from Editor ---
                        if edited_df is not None and not edited_df.equals(df_for_editor):
                            with st.spinner("Saving changes..."):
                                # Use the full dataset from session state for merging
                                current_df_list = st.session_state.transactions
                                current_df = pd.DataFrame(current_df_list)
                                if 'id' not in current_df.columns: current_df['id'] = [str(time.time() + i) for i in range(len(current_df))]
                                current_df['id'] = current_df['id'].astype(str); current_df.set_index('id', inplace=True)

                                # Prepare the edited part
                                edited_part_df = edited_df.copy()
                                if 'id' not in edited_part_df.columns:
                                    st.warning("ID lost, recovering..."); original_ids = df_for_editor['id'].astype(str).tolist()
                                    if len(original_ids) == len(edited_part_df): edited_part_df['id'] = original_ids
                                    else: st.error("Save failed: Mismatch."); st.stop()
                                edited_part_df['id'] = edited_part_df['id'].astype(str); edited_part_df.set_index('id', inplace=True)

                                # Update existing rows in the full DataFrame
                                current_df.update(edited_part_df)

                                # Handle deletions (if allowed)
                                if not is_large_dataset:
                                    editor_ids = set(edited_part_df.index); full_ids_after_update = set(current_df.index)
                                    deleted_ids = full_ids_after_update - editor_ids
                                    if deleted_ids: current_df = current_df.drop(index=list(deleted_ids))

            
                                if not is_large_dataset:
                                     new_rows_in_editor = edited_part_df[~edited_part_df.index.isin(current_df.index)]
                                     if not new_rows_in_editor.empty:
                                         st.warning("Saving new rows...")
                                         new_rows_dict = new_rows_in_editor.reset_index().to_dict('records')
                                         valid_new_rows = []
                                         for row in new_rows_dict: # Add missing required fields
                                             if 'User' not in row or pd.isna(row.get('User')): row['User'] = st.session_state.username
                                             if 'id' not in row or not row['id'] or pd.isna(row.get('id')): row['id'] = str(time.time() + random.random())
                                             if pd.notna(row.get('Date')): valid_new_rows.append(row)

                                         # Append directly to the session state list before converting back
                                         st.session_state.transactions.extend(valid_new_rows)
                                         # Recreate DataFrame from updated list if needed for consistency
                                         current_df = pd.DataFrame(st.session_state.transactions)
                                         current_df['id'] = current_df['id'].astype(str); current_df.set_index('id', inplace=True)


                                # Save updated data back to session state
                                st.session_state.transactions = current_df.reset_index().to_dict('records')
                                st.success("Changes saved!")
                                st.rerun() 

                     elif isinstance(df_for_editor, pd.DataFrame) and df_for_editor.empty and total_rows > 0 and is_large_dataset:
                          st.info(f"No transactions for page {page_number}.")
                     elif isinstance(df_for_editor, pd.DataFrame) and df_for_editor.empty and total_rows == 0:
                          st.info("No transactions to edit.")
                     else: st.error("Cannot display editor.")
            else: st.error("Internal data error: Cannot prepare editor.")
        except Exception as e: st.error(f"Editor error: {e}"); st.exception(e)

    else:
        st.info("Add transactions or upload a CSV to get started.")