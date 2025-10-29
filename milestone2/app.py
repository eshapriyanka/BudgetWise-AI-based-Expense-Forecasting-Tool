import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import random
import nltk 

# NLTK 
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt') # This looks for the data needed by word_tokenize
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True) 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Pre-load English stopwords
stop_words = set(stopwords.words('english'))

# Categoriize
CATEGORIES_KEYWORDS = {
    'Housing': {'rent', 'mortgage', 'property', 'tax', 'home', 'insurance', 'hoa', 'plumbing', 'electrician', 'repair', 'furniture', 'ikea', 'depot'},
    'Transportation': {'uber', 'car', 'lyft', 'taxi', 'bus', 'subway', 'amtrak', 'train', 'fuel', 'gasoline', 'payment', 'insurance', 'parking', 'wash', 'auto', 'repair', 'dmv', 'bolt'},
    'Groceries & Household': {'grocery', 'groceries', 'market', 'safeway', 'kroger', 'walmart', 'costco', 'sprouts', 'trader', 'joe', 'publix', 'food', 'supermarket', 'target', 'whole', 'foods', 'household', 'supplies', 'toilet', 'paper', 'soap', 'detergent'},
    'Dining': {'restaurant', 'cafe', 'coffee', 'snaks', 'starbucks', 'doordash', 'grubhub', 'ubereats', 'delivery', 'mcdonalds', 'burger', 'king', 'pizza', 'hut', 'dominos', 'chipotle', 'eats', 'eating', 'out'},
    'Entertainment': {'movie', 'cinema', 'concert', 'spotify', 'netflix', 'hulu', 'disney', 'app', 'store', 'google', 'play', 'tickets', 'bar', 'nightclub', 'apple', 'music', 'youtube', 'premium', 'gaming', 'steam', 'playstation', 'xbox'},
    'Personal & Family Care': {'haircut', 'shopping' ,'salon', 'barber', 'cosmetics', 'toiletries', 'sephora', 'ulta', 'gym', 'fitness', 'yoga', 'childcare', 'daycare', 'baby', 'pet', 'food', 'vet', 'veterinarian', 'spa', 'massage'},
    'Work & Education': {'office', 'supplies', 'stationery', 'udemy', 'coursera', 'book', 'textbook', 'tuition', 'school', 'college', 'webinar', 'software', 'adobe', 'slack', 'zoom', 'linkedin', 'learning', 'github'},
    'Health & Medical': {'doctor', 'dentist', 'hospital', 'pharmacy', 'cvs', 'walgreens', 'rite', 'aid', 'medicine', 'prescription', 'health', 'insurance', 'copay', 'vision', 'therapy', 'physician'},
    'Travel': {'flight', 'airline', 'american', 'delta', 'united', 'southwest', 'hotel', 'airbnb', 'booking.com', 'expedia', 'vacation', 'trip', 'luggage', 'rental', 'car', 'hertz', 'avis'},
    'Technology & Communication': {'phone', 'bill', 'verizon', 'at&t', 't-mobile', 'internet', 'comcast', 'xfinity', 'google', 'fi', 'gadget', 'apple', 'samsung', 'best', 'buy', 'newegg', 'aws', 'gcp', 'azure', 'domain'},
    'Financial & Insurance': {'life', 'insurance', 'bank', 'fee', 'atm', 'financial', 'advisor', 'investment', 'stock', 'coinbase', 'robinhood', 'loan', 'payment', 'student', 'credit', 'card', 'transfer'},
    'Business Expenses': {'client', 'dinner', 'business', 'travel', 'consulting', 'legal', 'fee', 'advertising', 'quickbooks', 'adwords'},
    'Taxes': {'tax', 'return', 'irs', 'property tax', 'income', 'prep', 'h&r', 'block', 'turbotax'},
    'Income': {'salary', 'paycheck', 'deposit', 'bonus', 'freelance', 'invoice', 'refund', 'reimbursement', 'interest', 'dividend'},
    'Other': {'charity', 'donation', 'gift'}
}
for category in CATEGORIES_KEYWORDS:
    CATEGORIES_KEYWORDS[category] = set(CATEGORIES_KEYWORDS[category])

# NLTK-based Categorization Function
def categorize_transaction_nltk(description, trans_type):
    if not isinstance(description, str) or not description.strip(): return 'Other'
    tokens = word_tokenize(description.lower())
    filtered_tokens = {word for word in tokens if word.isalpha() and word not in stop_words}
    if not filtered_tokens: return 'Other'
    if trans_type == "Income":
        if not filtered_tokens.isdisjoint(CATEGORIES_KEYWORDS['Income']): return 'Income'
        return 'Income'
    for category, keywords_set in CATEGORIES_KEYWORDS.items():
        if category == 'Income': continue
        if not filtered_tokens.isdisjoint(keywords_set): return category
    return 'Other'

# Processing for uploaded CSV (USING NLTK)
def process_uploaded_data(df_raw, username):
    print("\n Starting process_uploaded_data (NLTK version)")
    df = df_raw.copy()
    date_col_original = None
    if 'Date' in df.columns: date_col_original = 'Date'
    elif 'date' in df.columns: date_col_original = 'date'
    if date_col_original:
        initial_rows = len(df); df.dropna(subset=[date_col_original], inplace=True)
        if pd.api.types.is_string_dtype(df[date_col_original]): df = df[df[date_col_original].str.strip() != '']
        rows_dropped = initial_rows - len(df); print(f"DEBUG: Dropped {rows_dropped} rows missing date.")
        if df.empty: st.error("No rows with dates found."); return None
    else: st.error("Cannot find 'Date' column."); return None
    original_category_col_name = None
    if 'Category' in df.columns: original_category_col_name = 'category_original'; df.rename(columns={'Category': original_category_col_name}, inplace=True)
    elif 'category' in df.columns: original_category_col_name = 'category_original'; df.rename(columns={'category': original_category_col_name}, inplace=True)
    df.columns = [str(col).lower().strip() for col in df.columns]
    required_cols = ['date', 'amount']
    if not all(col in df.columns for col in required_cols): st.error(f"Missing required columns. Found: {list(df.columns)}"); return None
    try:
        df['date_converted'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        nat_count = df['date_converted'].isnull().sum(); st.session_state.invalid_date_count = nat_count
        if nat_count > 0: st.warning(f"{nat_count} invalid dates removed.")
        df['date'] = df['date_converted']; df.dropna(subset=['date'], inplace=True)
        if df.empty: st.warning("All rows removed due to invalid dates."); return None
    except Exception as e: st.error(f"Date conversion error: {e}"); return None
    df['Type'] = 'Expense';
    if 'type' in df.columns: df.loc[df['type'].str.contains('income', case=False, na=False), 'Type'] = 'Income'
    elif 'type' not in df.columns: df.loc[df['amount'] > 0, 'Type'] = 'Income'
    df['amount'] = df['amount'].abs()
    if 'description' not in df.columns: df['description'] = ""
    df['description'] = df['description'].fillna("").astype(str)
    originally_blank_description = df['description'].str.strip() == ""
    print("DEBUG: Applying NLTK categorization..."); start_time = time.time()
    # Apply the NLTK function row-by-row
    df['category'] = df.apply(lambda row: categorize_transaction_nltk(row['description'], row['Type']), axis=1)
    end_time = time.time(); print(f"DEBUG: NLTK categorization took {end_time - start_time:.2f} seconds.")
    if original_category_col_name is not None and original_category_col_name in df.columns:
         valid_original = df[original_category_col_name].notna() & (df[original_category_col_name].astype(str).str.strip() != '')
         df.loc[originally_blank_description & valid_original & (df['category'] == 'Other'), 'category'] = df[original_category_col_name]
         df['category'] = df['category'].astype(str).str.strip()
    blank_desc_mask = df['description'].str.strip() == ""
    valid_category_mask = (df['category'] != 'Other') & (df['category'] != '') & (df['category'].notna())
    df.loc[blank_desc_mask & valid_category_mask, 'description'] = "Uploaded: " + df['category']
    df.loc[df['description'].str.strip() == "", 'description'] = "Uploaded Data"
    df.rename(columns={'date': 'Date', 'amount': 'Amount', 'description': 'Description', 'category': 'Category'}, inplace=True)
    df['id'] = [f"csv_{time.time()}_{i}" for i in range(len(df))]
    df['User'] = username
    final_cols = ['id', 'Date', 'Type', 'Amount', 'Description', 'Category', 'User']
    df_final = df[[col for col in final_cols if col in df.columns]].copy()
    print("--- End process_uploaded_data ---\n")
    return df_final.to_dict('records')

# Initialize Session State Variables
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'transactions' not in st.session_state: st.session_state.transactions = []
if 'uploaded_file_processed' not in st.session_state: st.session_state.uploaded_file_processed = False
if 'current_uploaded_file' not in st.session_state: st.session_state.current_uploaded_file = None
if 'invalid_date_count' not in st.session_state: st.session_state.invalid_date_count = 0

# Page Config & Custom CSS
st.set_page_config(page_title="BudgetWise", layout="wide")
st.markdown("""<style>.stApp { background: linear-gradient(to right top, #d3e0ff, #e4eaff, #f5f5ff, #ffffff, #ffffff); } h1, h2, h3 { color: #0d47a1; } .st-emotion-cache-91n1wr p { font-weight: bold; color: #1565c0; } </style>""", unsafe_allow_html=True)

# Authentication Page
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

# Main Application
else:
    # --- Sidebar ---
    st.sidebar.title(f"Welcome, {st.session_state.username}!")
    def handle_logout():
        keys_to_keep = []
        for key in list(st.session_state.keys()):
             if key not in keys_to_keep: del st.session_state[key]
        st.rerun()
    st.sidebar.button("Logout", on_click=handle_logout)
    st.sidebar.markdown("---"); st.sidebar.info("BudgetWise")

    # Main Page Title
    st.title("Budgetwise")

    # File Uploader Section
    with st.expander("Upload Expense History (CSV)", expanded=True):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key='file_uploader_widget')
        if uploaded_file is not None:
            if uploaded_file != st.session_state.current_uploaded_file or not st.session_state.uploaded_file_processed:
                st.session_state.current_uploaded_file = uploaded_file
                st.session_state.invalid_date_count = 0
                try:
                    with st.spinner('Reading and processing file using NLTK...'):
                        df_read = pd.read_csv(uploaded_file, low_memory=False)
                        print(f"DEBUG TERMINAL: pd.read_csv initial shape: {df_read.shape}")
                        new_data = process_uploaded_data(df_read, st.session_state.username)
                    if new_data is not None:
                        st.session_state.transactions = new_data
                        st.session_state.uploaded_file_processed = True
                        st.success(f"Successfully processed {len(new_data)} transactions!")
                        if st.session_state.invalid_date_count > 0: st.warning(f"{st.session_state.invalid_date_count} rows removed due to invalid dates.")
                        st.rerun()
                    else:
                        st.session_state.current_uploaded_file = None; st.session_state.uploaded_file_processed = False; st.session_state.transactions = []
                except Exception as e:
                    st.error(f"Error during file processing: {e}"); st.exception(e)
                    st.session_state.current_uploaded_file = None; st.session_state.uploaded_file_processed = False; st.session_state.transactions = []
        elif st.session_state.current_uploaded_file is not None:
             st.info("File removed. Clearing data.")
             st.session_state.current_uploaded_file = None; st.session_state.uploaded_file_processed = False; st.session_state.transactions = []; st.session_state.invalid_date_count = 0
             st.rerun()

    # Manual Transactions
    st.header("Log your Transactions")
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
            category = categorize_transaction_nltk(description, trans_type) # Use NLTK here too
            new_transaction = { "id": str(time.time()), "Date": trans_date, "Type": trans_type, "Amount": amount, "Description": description, "Category": category, "User": st.session_state.username }
            st.session_state.transactions.append(new_transaction)
            st.success(f"Added {trans_type}: ${amount} (Category: {category})")
            st.rerun()

    st.markdown("---")

    # Transaction Editor Section
    st.header("Manage your Transactions")
    try:
        if isinstance(st.session_state.transactions, list) and all(isinstance(item, dict) for item in st.session_state.transactions):
             all_data_df = pd.DataFrame(st.session_state.transactions)
             if 'id' not in all_data_df.columns: all_data_df['id'] = [str(time.time() + i) for i in range(len(all_data_df))]
             all_data_df['id'] = all_data_df['id'].astype(str)
             if 'Date' in all_data_df.columns:
                 all_data_df['Date'] = pd.to_datetime(all_data_df['Date'], errors='coerce')
                 all_data_df.dropna(subset=['Date'], inplace=True)
                 all_data_df = all_data_df.sort_values(by="Date", ascending=False, na_position='last').reset_index(drop=True)
             #else:
                  #st.warning("Sorting disabled: 'Date' missing."); all_data_df = all_data_df.reset_index(drop=True)

             ROWS_PER_PAGE = 50; total_rows = len(all_data_df); is_large_dataset = total_rows > ROWS_PER_PAGE * 10
             if total_rows == 0: st.info("No transactions to edit."); df_for_editor = pd.DataFrame()
             else:
                 if is_large_dataset:
                     total_pages = max(1, (total_rows // ROWS_PER_PAGE) + (1 if total_rows % ROWS_PER_PAGE > 0 else 0))
                     page_number = st.slider("Page", 1, total_pages, 1)
                     start_idx = (page_number - 1) * ROWS_PER_PAGE; end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
                     start_idx = max(0, start_idx); end_idx = min(total_rows, end_idx)
                     df_for_editor = all_data_df.iloc[start_idx:end_idx] if start_idx < end_idx else pd.DataFrame(columns=all_data_df.columns)
                 else:
                     st.info("Double-click to edit. Select rows & press Delete.")
                     df_for_editor = all_data_df
                 all_categories = list(CATEGORIES_KEYWORDS.keys())
                 if isinstance(df_for_editor, pd.DataFrame) and not df_for_editor.empty:
                    edited_df = st.data_editor(df_for_editor, key="data_editor", num_rows="dynamic" if not is_large_dataset else "fixed", disabled=["User", "id"],
                        column_config={"id": None, "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),"Amount": st.column_config.NumberColumn("Amount", format="$%.2f", required=True),"Category": st.column_config.SelectboxColumn("Category", options=all_categories, required=True),"Type": st.column_config.SelectboxColumn("Type", options=["Expense", "Income"], required=True),"Description": st.column_config.TextColumn("Description", required=False)}, width='stretch', hide_index=True,)
                    if edited_df is not None and not edited_df.equals(df_for_editor):
                        with st.spinner("Saving changes..."):
                            current_df = pd.DataFrame(st.session_state.transactions)
                            if 'id' not in current_df.columns: current_df['id'] = [str(time.time() + i) for i in range(len(current_df))]
                            current_df['id'] = current_df['id'].astype(str); current_df.set_index('id', inplace=True)
                            edited_part_df = edited_df.copy()
                            if 'id' not in edited_part_df.columns:
                                st.warning("ID lost, recovering..."); original_ids = df_for_editor['id'].astype(str).tolist()
                                if len(original_ids) == len(edited_part_df): edited_part_df['id'] = original_ids
                                else: st.error("Save failed: Mismatch."); st.stop()
                            edited_part_df['id'] = edited_part_df['id'].astype(str); edited_part_df.set_index('id', inplace=True)
                            # Re-categorize edited rows using NLTK
                            ids_to_recategorize = edited_part_df[edited_part_df['Description'] != df_for_editor.loc[edited_part_df.index, 'Description']].index
                            if not ids_to_recategorize.empty:
                                print(f"DEBUG: Re-categorizing {len(ids_to_recategorize)} edited rows using NLTK...")
                                edited_part_df.loc[ids_to_recategorize, 'Category'] = edited_part_df.loc[ids_to_recategorize].apply(lambda row: categorize_transaction_nltk(row['Description'], row['Type']), axis=1)
                            current_df.update(edited_part_df)
                            if not is_large_dataset:
                                editor_ids = set(edited_part_df.index); full_ids_after_update = set(current_df.index)
                                deleted_ids = full_ids_after_update - editor_ids
                                if deleted_ids: current_df = current_df.drop(index=list(deleted_ids))
                                new_rows_in_editor = edited_part_df[~edited_part_df.index.isin(current_df.index)]
                                if not new_rows_in_editor.empty:
                                     st.warning("Saving new rows...")
                                     new_rows_dict = new_rows_in_editor.reset_index().to_dict('records')
                                     valid_new_rows = []
                                     for row in new_rows_dict:
                                         if 'User' not in row or pd.isna(row.get('User')): row['User'] = st.session_state.username
                                         if 'id' not in row or not row['id'] or pd.isna(row.get('id')): row['id'] = str(time.time() + random.random())
                                         if pd.notna(row.get('Date')):
                                              row['Category'] = categorize_transaction_nltk(row.get('Description', ''), row.get('Type', 'Expense')) # Categorize new row
                                              valid_new_rows.append(row)
                                     temp_list = current_df.reset_index().to_dict('records'); temp_list.extend(valid_new_rows)
                                     st.session_state.transactions = temp_list
                                     current_df = pd.DataFrame(st.session_state.transactions)
                                     current_df['id'] = current_df['id'].astype(str); current_df.set_index('id', inplace=True)
                            st.session_state.transactions = current_df.reset_index().to_dict('records')
                            st.success("Changes saved!")
                            st.rerun()
                 elif isinstance(df_for_editor, pd.DataFrame) and df_for_editor.empty and total_rows > 0 and is_large_dataset: st.info(f"No transactions for page {page_number}.")
                 elif isinstance(df_for_editor, pd.DataFrame) and df_for_editor.empty and total_rows == 0: st.info("No transactions to edit.")
                 else: st.error("Cannot display editor.")
        else: st.error("Internal data error: Cannot prepare editor.")
    except Exception as e: st.error(f"Editor error: {e}"); st.exception(e)

    st.markdown("---")

    # Analystics
    st.header("Your Financial Analysis")
    if st.session_state.transactions:
        with st.spinner("Analyzing data & building charts..."):
            time.sleep(0.1)
            if not isinstance(st.session_state.transactions, list) or not all(isinstance(item, dict) for item in st.session_state.transactions): st.error("Internal data error."); st.stop()
            df = pd.DataFrame(st.session_state.transactions)
            if df.empty: st.info("No transactions to analyze."); st.stop()
            try:
                if 'Date' not in df.columns: raise ValueError("'Date' column missing")
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                null_dates_final = df['Date'].isnull().sum()
                if null_dates_final > 0: df.dropna(subset=['Date'], inplace=True)
                if df.empty: raise ValueError("No valid dates remaining after final check.")
            except Exception as e: st.error(f"Date error: {e}"); st.stop()
            expenses_df = df[df['Type'] == 'Expense'].copy()

            # Metrics
            st.subheader("Income vs. Expense")
            total_income=df[df['Type']=='Income']['Amount'].sum(); total_expense=expenses_df['Amount'].sum(); net_savings=total_income-total_expense
            col1,col2,col3=st.columns(3); col1.metric("Total Income",f"${total_income:,.2f}"); col2.metric("Total Expense",f"${total_expense:,.2f}"); col3.metric("Net Savings",f"${net_savings:,.2f}",delta=f"{net_savings:,.2f}",delta_color="normal" if net_savings >= 0 else "inverse")

            # Charts
            if not expenses_df.empty:
                st.subheader("Expense Analysis")
                col1, col2 = st.columns(2)
                with col1: # Pie Chart
                    st.write("Spending by Category"); category_spending=expenses_df.groupby('Category')['Amount'].sum()
                    if not category_spending.empty:
                        value_threshold=category_spending.sum()*0.02; small_sum=category_spending[category_spending<value_threshold].sum(); main=category_spending[category_spending>=value_threshold]
                        if small_sum>0 and not main.empty: main['Other (<2%)']=small_sum
                        elif small_sum>0: main=pd.Series({'Other (<2%)': small_sum})
                        if not main.empty:
                            fig,ax=plt.subplots(figsize=(10,8)); palette=sns.color_palette('pastel',len(main))
                            try: ax.pie(main, labels=main.index, autopct='%1.1f%%',startangle=90,colors=palette); ax.axis('equal'); st.pyplot(fig)
                            except Exception as e: st.error(f"Pie chart error: {e}")
                        else: st.info("No data for pie.")
                    else: st.info("No category spending.")
                with col2: # Bar Chart
                    st.write("Spending Summary"); category_df=expenses_df.groupby('Category')['Amount'].sum().reset_index(name="Amount")
                    if not category_df.empty: st.bar_chart(category_df,x="Category",y="Amount")
                    else: st.info("No category spending.")
                # Monthly Line Chart
                st.subheader("Monthly Spending")
                try:
                    dated=expenses_df.set_index('Date')
                    if isinstance(dated.index, pd.DatetimeIndex):
                        monthly=dated.resample('ME')['Amount'].sum();
                        if not monthly.empty: st.line_chart(monthly)
                        else: st.info("No monthly data.")
                    else: st.warning("Date index issue.")
                except Exception as e: st.error(f"Monthly chart error: {e}")
                # Yearly Table
                st.subheader("Monthly Breakdown (Yearly)")
                try:
                    if pd.api.types.is_datetime64_any_dtype(expenses_df['Date']):
                        expenses_df['Year']=expenses_df['Date'].dt.year; years=sorted(expenses_df['Year'].unique(),reverse=True)
                        if years:
                            year=st.selectbox("Year:", years); yearly=expenses_df[expenses_df['Year']==year]
                            if not yearly.empty:
                                yearly=yearly.copy(); yearly['Month']=yearly['Date'].dt.strftime('%Y-%m'); summary=yearly.pivot_table(index='Category',columns='Month',values='Amount',aggfunc='sum',fill_value=0); st.dataframe(summary)
                            else: st.info(f"No data for {year}.")
                        else: st.info("No yearly data.")
                    else: st.warning("Cannot extract year.")
                except Exception as e: st.error(f"Yearly breakdown error: {e}")
            else:
                st.info("Add expenses.")

    # if no transactions exist in the session state
    else:
        st.info("Add transactions or upload a CSV to get started.")

