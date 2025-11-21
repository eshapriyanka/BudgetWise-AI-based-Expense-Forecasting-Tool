# BudgetWise

**BudgetWise** is an intelligent, data-driven personal finance application designed to help individuals track expenses, forecast future spending, and manage financial goals. Built with **Streamlit**, it leverages **Natural Language Processing (NLTK)** for automated transaction categorization and **Facebook Prophet** for predictive expense forecasting.

---

## Key Features

### Secure Authentication
* **Role-Based Access Control (RBAC):** Distinct dashboards for standard Users and Administrators.
* **Secure Login:** Password hashing (SHA-256) for secure credential storage.
* **Persistent Sessions:** User data remains secure and accessible throughout the session.

### AI-Powered Intelligence
* **Automated Categorization:** Uses **NLTK** (Natural Language Toolkit) to analyze transaction descriptions and automatically assign categories (e.g., "Uber" $\rightarrow$ "Transportation").
* **Predictive Forecasting:** Integrated **Prophet** model forecasts future spending trends up to 365 days in advance based on historical data.

### Interactive Dashboards
* **Financial Snapshot:** Real-time metrics for Income, Expenses, and Net Savings.
* **Visual Analytics:** Interactive charts (Altair & Matplotlib) showing spending breakdown by category, monthly trends, and yearly comparisons.
* **Transaction Editor:** A smart, paginated spreadsheet interface to view, edit, or delete transactions.

### Goal Setting & Tracking
* **Budget Management:** Set monthly spending limits for specific categories (e.g., Dining, Shopping).
* **Real-Time Alerts:** Visual progress bars turn red when you exceed your budget limit.

### Admin Control Panel
* **System Monitoring:** View total registered users, active goals, and transaction volume.
* **Dynamic Categories:** Add new categories or keywords instantly to the database to improve the AI categorization engine for all users without changing code.

---

## System Architecture

The application follows a modular architecture separating the user interface, logic processing, and data storage.

```mermaid
graph TD
    User([User/Admin]) --> Login[Login Interface]
    Login -- Authenticate --> DB[(SQLite Database)]
    Login --> Router{Role?}
    
    Router -- Admin --> AdminDash[Admin Dashboard]
    Router -- User --> UserDash[User Dashboard]
    
    subgraph "User Features"
        UserDash --> Upload[CSV Upload]
        UserDash --> Manual[Manual Entry]
        UserDash --> Analysis[Financial Analysis]
        UserDash --> Forecast[Forecasting Engine]
    end
    
    subgraph "Processing Layer"
        Upload --> NLTK[[NLTK Engine]]
        Manual --> NLTK
        NLTK -- Categorized Data --> DB
        DB --> Prophet[[Prophet AI Model]]
        Prophet --> Forecast
    end
    
    subgraph "Admin Features"
        AdminDash --> ManageCat[Manage Keywords]
        ManageCat -- Update Rules --> DB
    end
