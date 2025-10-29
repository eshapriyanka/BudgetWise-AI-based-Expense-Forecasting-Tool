# Project Documentation: BudgetWise AI-Based Expense Forecasting Tool

## 1.0 Introduction
The BudgetWise AI-Based Expense Forecasting Tool is a smart, data-driven application designed to help individuals and businesses manage their finances proactively. The tool tracks user expenses, automatically categorizes them, and leverages Artificial Intelligence (AI) and Machine Learning (ML) models to forecast future spending patterns. By providing predictive insights and actionable alerts, it empowers users to maintain their budgets, anticipate overspending, and make informed financial decisions.

## 2.0 Problem Statement
Effective budget management is a common challenge for many. The core problems this project aims to address are:
* Unconscious Overspending: Many people overspend without a clear, real-time understanding of their financial habits.
* Inefficiency of Manual Budgeting: Traditional methods of manually tracking expenses are often time-consuming, tedious, and prone to human error.
* Difficulty in Future Planning: Accurately estimating future expenses is difficult, making long-term financial planning and savings a significant challenge.

## 3.0 Outcomes
* Clear Financial Overview: Provide users with an easy-to-understand dashboard of their income, expenses, and savings.
* Automated Expense Forecasting: Predict future spending based on historical data, helping users anticipate financial needs.
* Spending Pattern Identification: Automatically categorize transactions and highlight key spending areas.
* Goal-Oriented Planning: Assist users in setting and tracking progress towards financial goals (e.g., saving for a down payment, retirement).
* Data-Driven Insights: Empower users to identify areas for potential savings and improve budgeting habits.
* User-Friendly Interface: An intuitive platform for inputting transactions, viewing reports, and interacting with forecasts.

## 4.0 Modules to be implemented
* User Authentication & Profile Management:
  * User registration, login(standard email/pass)
  * Basic user profile for managing financial data and preferences
* Transaction Ingestion & Categorization Module:
  * Interface for users to manually input or upload simulated/dummy transaction data (ee.g., CSV).
  * Automated (or semi-automated, rule-based) categorization of transactions (e.g., 'Groceries', 'Utilities', 'Transport').
* Data Analysis & Reporting Module:
  * Calculate spending summaries per category, month, or custom period.
  * Generate reports on income vs. expenses.
* Forecasting Module:
  * Implement Prophet (Meta's forecasting library) to predict future expenses and income based on historical transaction patterns.
  * Allow users to define financial goals and forecast their achievement.
* Visualization & Dashboard Module:
  * Interactive charts and graphs (using Matplotlib, Seaborn) to visualize spending trends, forecasts, and goal progress.
  * A central dashboard providing a holistic financial overview.
* Admin Dashboard:
  * Management of transaction categories.
  * Monitoring of system usage and data integrity.

## 5.0 Week-wise module implementation and high-level requirements:
### Milestone 1: Weeks 1-2 - Foundation and Basic Input
This milestone focused on setting up the basic application structure, user interaction, and data entry capabilities.
#### Module 1: User Authentication & Basic Transaction Input

* User Authentication (Mock Implementation):
  * Requirement: Implement secure user registration and login.
  * Implementation: For this initial phase, a mock authentication system was implemented using Streamlit's st.session_state.
       * A simple login screen prompts for a username and password.
       * Basic validation (checking against a hardcoded password like "123") is performed.
       * Upon successful "login," a boolean flag st.session_state.logged_in is set to True, and the username is stored in st.session_state.username. The main application interface is conditionally displayed only when logged_in is True.
       * A "Logout" button resets the logged_in flag and clears session data.
  * Note: This mock system serves demonstration purposes. A production-ready version would require integrating a proper database for user credentials and implementing secure password hashing and token-based authentication (like JWT, as initially planned).
* Profile Management (Minimal):
  * Requirement: Create user profiles for managing financial data.
  * Implementation: Profile management is currently minimal. The logged-in st.session_state.username is used to display a welcome message and could theoretically be used to associate transactions with users if data persistence were implemented. No separate profile page or settings were built in this phase.
* Manual Transaction Input:
  * Requirement: Design a basic web interface for manual transaction input.
  * Implementation: A user-friendly form was created using st.form("transaction_form").
     * Input Widgets: Standard Streamlit widgets capture transaction details:
         * st.date_input for the date.
         * st.number_input for the amount (with minimum value validation).
         * st.selectbox for the transaction type ('Expense' or 'Income').
         * st.text_input for the description.
     * Data Handling: Upon form submission (st.form_submit_button):
         * Input validation checks if the description is provided.
         * The categorize_transaction function (detailed in Milestone 2) is called to determine the category based on the description.
         * A Python dictionary representing the new transaction (including date, amount, type, description, category, user, and a unique ID based on time.time()) is created.
         * This dictionary is appended to the st.session_state.transactions list, which acts as the in-memory data store for the current session.
         * st.rerun() is called to refresh the UI and display the updated data.
  

### Milestone 2: Weeks 3-4 - Data Processing, Analysis & Visualization
This milestone integrated Natural Language Processing (NLP) for categorization and provided data analysis capabilities.
#### Module 2: Transaction Categorization & Basic Reporting

  * Automated Transaction Categorization:
       *  Requirement: Implement a rule-based or simple NLP system using NLTK for automatic categorization.
       * Implementation:
           * NLTK Setup: The application imports the nltk library and ensures the necessary data packages (stopwords corpus and punkt tokenizer) are downloaded. English stopwords are loaded into a set for efficient filtering.
           * Keyword Dictionary: The CATEGORIES_KEYWORDS dictionary is maintained, mapping category names to sets of relevant single keywords (optimized for token matching).
           * categorize_transaction_nltk Function: This core function performs categorization:
             * It takes the description and trans_type.
             * Tokenization: The description is converted to lowercase and split into individual words (tokens) using nltk.word_tokenize.
             * Filtering: Punctuation and common English words (stopwords from nltk.corpus.stopwords) are removed, leaving only potentially meaningful keywords.
             * Keyword Matching: The remaining tokens are compared against the keyword sets defined in CATEGORIES_KEYWORDS. Set intersection (isdisjoint) is used for efficient matching. The first category whose keyword set overlaps with the transaction's tokens is assigned. Income transactions are handled as a special case. If no keywords match, the category defaults to 'Other'.
           * Application:
              * This NLTK function is called when a user submits a manual transaction via the form.
              * For CSV uploads, the process_uploaded_data function uses Pandas' df.apply() method to execute categorize_transaction_nltk on each row of the uploaded data, processing the 'description' and 'Type' columns. Note: This row-by-row NLTK application is less performant on large files than pure Pandas string methods but fulfills the NLTK requirement.
              * A fallback mechanism uses the original category from the CSV if the NLTK process results in 'Other' and the original description was blank. Blank descriptions are also filled post-categorization (e.g., "Uploaded: Dining"). 
           * Manual Override: The st.data_editor in the "Manage Transactions" section allows users to manually change the category assigned by NLTK using a dropdown, providing the override capability.
  * Spending Summary Reports & Analysis:
    * Requirement: Generate reports on spending, monthly summaries, and income vs. expenses using Pandas.
    * Implementation:
      * Data Preparation: Creates Pandas DataFrames (df, expenses_df) from st.session_state.transactions. Dates are converted/validated.
      * Income vs. Expense: Calculates totals and net savings, displayed using st.metric.
      * Monthly Spending (All Time): Uses Pandas resample('ME') to aggregate total monthly expenses, displayed with st.line_chart.
      * Monthly Breakdown (By Year): Includes a st.selectbox for year selection. Filters the data for the chosen year and displays a pivot_table (categories vs. months) using st.dataframe.         
  * Dashboard View:
    * Implementation:
      * Uses st.columns for layout.
      * Pie Chart: Generated using Matplotlib/Seaborn showing category spending percentages (small slices grouped). Displayed via st.pyplot.
      * Bar Chart: Uses st.bar_chart for an alternative view of category spending totals.
      * Dynamic Updates: Visualizations refresh automatically when transaction data changes.

### Milestone 3: Weeks 5-6
#### Module 3: Forecasting Engine & Goal Setting

* High-Level Requirements:

  * Historical Data Preparation: Prepare historical transaction data in the format required by the forecasting model (e.g., time series of aggregated daily/weekly/monthly expenses per category).
  * Prophet Integration: Integrate Prophet (Meta's forecasting library) to generate future expense forecasts for different categories or overall spending.
  * Financial Goal Setting: Allow users to define financial goals (e.g., "Save $X by Y date," "Reduce spending in category Z by A%").
  * Forecast Visualization: Update the dashboard to display the forecasted expenses alongside actual historical data (line chart using Matplotlib/Seaborn). Show projected savings based on forecasts.

-----------------

## 3.0 Objectives
To address the problems stated above, the project leverages AI to create a more dynamic and intelligent budgeting experience. The primary objectives are:
* Learn User Behavior: To develop a system that automatically learns and understands individual spending habits from historical data.
* Predict Future Expenses: To accurately forecast future expenses across different categories, enabling users to plan ahead.
* Provide Proactive Alerts: To issue real-time budget alerts to users when they are at risk of overspending in a particular category.

## 4.0 Project Scope
The scope of the project is defined by the flow of data from input to output.
* Input: The system will accept expense data from various sources, such as uploaded CSV files or through API integration with bank feeds or Google Sheets.
* Processing: The core of the tool involves pre-processing the data, performing feature engineering, and applying a suite of AI/ML models to analyze and forecast spending patterns.
* Output: The final output will be delivered to the user through an interactive dashboard that provides spending forecasts, budget alerts, and actionable financial insights.

## 5.0 Methodology and System Architecture
The project follows a structured lifecycle, progressing from data handling to model deployment;
* Data Ingestion & Pre-processing: Acquiring and cleaning expense data.
* Exploratory Data Analysis (EDA): Understanding patterns and trends in the data.
* Feature Engineering: Creating relevant features to improve model performance.
* Machine Learning Modeling: Implementing and training various forecasting models.
* Deep Learning Modeling: Utilizing advanced models like LSTMs for complex patterns.
* Dashboard & Visualization: Building a user-facing application to display results.
* Deployment: Hosting the application on a cloud platform.

## 5.1 Feature Engineering
To enhance model accuracy, several features will be engineered from the raw data:
* Time-Based Features: Extracting components like the month, day of the week, and seasonality.
* Behavioral Features: Calculating cumulative spending patterns to understand user habits over time.
* Categorical Features: Grouping expenses into logical categories (e.g., groceries, transport, entertainment).

## 6.0 Forecasting Models
The tool will implement and compare several forecasting models to ensure the highest accuracy.

### 6.1 Time Series Models: ARIMA/SARIMA
Time series models are ideal for data with clear temporal trends.
They are effective at capturing seasonality and cyclical patterns in spending data.
Specific models to be implemented include ARIMA (Autoregressive Integrated Moving Average) and SARIMA (Seasonal ARIMA).

### 6.2 Prophet Model
Developed by Facebook, Prophet is a powerful and user-friendly forecasting model.
It is robust in handling seasonality and holidays, which are common in expense data.
It offers a simple API and produces fast, reliable results.

### 6.3 Deep Learning Models: RNNs & LSTMs
For more complex and long-term forecasting, advanced deep learning techniques will be used.
Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are specifically designed for sequential data like monthly expenses.
These models excel at capturing long-term dependencies in spending habits that simpler models might miss.

### 6.4 Anomaly Detection
The system will also include a module to identify unusual spending activities. This helps users quickly spot potential fraudulent transactions or significant deviations from their normal budget, such as a sudden large purchase.

## 7.0 Model Evaluation
The performance of all forecasting models will be rigorously evaluated using standard statistical metrics:
Mean Absolute Error (MAE): Measures the average magnitude of the errors in a set of predictions, without considering their direction.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Root Mean Square Error (RMSE): The square root of the average of squared differences between prediction and actual observation. It gives a higher weight to large errors. 

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Mean Absolute Percentage Error (MAPE): Expresses the accuracy as a percentage of the error. 

$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

## 8.0 Application Layer and Features
The user-facing application will be a key component, providing an intuitive interface to interact with the AI-powered insights.

### 8.1 Dashboard Overview
The main dashboard will provide a comprehensive summary of the user's finances:
* Visualization of total spending over time.
* A breakdown of spending distribution by category.
* An interactive graph displaying future expense forecasts.

### 8.2 Budget Alerts
The system will send automated alerts via push notifications or emails to warn users. For example: “⚠️ You are on track to overshoot your travel budget by 20% this month”.

### 8.3 API Integration
To ensure data is always up-to-date, the tool will support integration with:
* Google Sheets: for automatic data synchronization.
* Bank Feeds: to pull in real-time transaction data.

### 8.4 Scenario Analysis
An advanced feature will allow users to perform "what-if" analysis. Users can simulate the impact of financial changes (e.g., “What if my rent increases by 10%?”) and the AI will forecast the effect on their overall budget.


## 9.0 Technology Stack
Programming Language: Python
* AI/ML Libraries: Scikit-learn, Statsmodels (for ARIMA), Prophet, TensorFlow/Keras (for LSTM)
* Data Visualization: Matplotlib, Seaborn, and Plotly (for interactive charts).
* Web App Frameworks: Streamlit (for rapid dashboard development), Flask, or Dash.
* Cloud Deployment: Heroku, Render, AWS, or GCP.

## 10.0 Security and Compliance
Handling sensitive financial data requires a strong focus on security.
* All financial data will be encrypted both in transit and at rest.
* User privacy will be a priority, with strict data handling protocols.
* The system will be designed with considerations for financial regulations like GDPR.

## 11.0 Future Enhancements
The project has a clear roadmap for future development:
* NLP for Transaction Categorization: Use Natural Language Processing to automatically categorize expenses from transaction descriptions.
* Third-Party Integration: Integrate with popular personal finance apps (like Mint or YNAB) for a more connected experience.
* AI Chat Assistant: Develop an AI-powered chatbot to provide users with budgeting advice and answer financial questions.

## 12.0 Key Challenges and Best Practices
### 12.1 Challenges
* Data Quality: Ensuring the input expense data is clean and consistent is crucial for model accuracy.
* Model Accuracy: Continuously tuning and validating models to maintain high forecasting accuracy.
* User Adoption: Designing an intuitive and valuable user experience to encourage consistent use.

### 12.2 Best Practices
* Interpretability: Keep models as interpretable as possible so users can understand the basis of the forecasts.
* Iterative Development: Start with simple, effective models (like ARIMA or Prophet) and progressively add complexity (like LSTMs).
* Usability Focus: The primary focus will be on creating a usable and helpful tool for the end-user.


## Project Timeline and Milestones
### Milestone_1
#### Week 1 : 22 Sept, 2025 to 26 Sept, 2025
* Covered foundational AI/ML concepts relevant to the project.
* Received an in-depth introduction to the project scope, goals, and real-world applications.

#### Week 2 : 29 Sept, 2025 to 3 Oct, 2025 
* Focused on practical learning and upskilling.
* Completed tutorials for Python, covering basic data structures and programming concepts.
* Gained practical experience with core ML libraries, including Pandas for data manipulation, NumPy for numerical operations and data visualizations.

#### Week 3 : 6 Oct, 2025 to 10 Oct, 2025
* Successfully sourced and finalized the dataset required for the forecasting model.
* Researched and confirmed the technology stack for the project (Python, Streamlit, Statsmodels, etc.).
* Initiated the first phase of development, including data cleaning, preprocessing, and the initial setup for model training.
