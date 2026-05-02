"""Generate realistic sample datasets for the Autonomous Data Analyst Agent.

Datasets:
  - sales_data.csv      : 3-year monthly sales with seasonal patterns + anomalies
  - ecommerce.csv       : 8k e-commerce orders with returns, discounts, trends
  - hr_analytics.csv    : 1.5k employee records with attrition patterns

Run:  python data/generate_sample_data.py
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

OUTPUT_DIR = Path(__file__).parent / "sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  Sales Data
# ---------------------------------------------------------------------------

def generate_sales_data(n: int = 5000) -> pd.DataFrame:
    regions = ["North", "South", "East", "West", "Central"]
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Food & Beverage"]
    products = {
        "Electronics": ["Laptop Pro", "Wireless Earbuds", "Smart Watch", "Tablet"],
        "Clothing": ["Winter Jacket", "Running Shoes", "Denim Jeans", "T-Shirt Pack"],
        "Home & Garden": ["Coffee Maker", "Robot Vacuum", "Garden Kit", "Air Purifier"],
        "Sports": ["Yoga Mat", "Dumbbell Set", "Bike Helmet", "Swim Goggles"],
        "Food & Beverage": ["Protein Powder", "Organic Tea", "Energy Bars", "Coffee Blend"],
    }
    channels = ["Online", "Retail Store", "Partner", "Direct Sales"]
    salespersons = [f"SP_{i:03d}" for i in range(1, 31)]

    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2024-12-31")
    date_range = pd.date_range(start, end, freq="D")

    rows = []
    for _ in range(n):
        date = pd.Timestamp(rng.choice(date_range))
        region = rng.choice(regions)
        category = rng.choice(categories)
        product = rng.choice(products[category])
        channel = rng.choice(channels)
        salesperson = rng.choice(salespersons)

        # Seasonal multiplier: Q4 boost, Q3 dip
        month = date.month
        seasonal = 1.0 + 0.3 * np.sin((month - 3) * np.pi / 6)

        # Region-specific performance: West underperforms in 2024
        region_factor = 0.7 if (region == "West" and date.year == 2024) else 1.0

        base_price = {
            "Electronics": 450, "Clothing": 65, "Home & Garden": 120,
            "Sports": 55, "Food & Beverage": 30,
        }[category]

        unit_price = base_price * rng.uniform(0.8, 1.3)
        quantity = int(rng.integers(1, 20) * seasonal * region_factor)
        revenue = round(unit_price * quantity, 2)
        profit_margin = round(rng.uniform(0.15, 0.55), 3)

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "year": date.year,
            "month": date.month,
            "quarter": f"Q{(date.month - 1) // 3 + 1}",
            "region": region,
            "category": category,
            "product_name": product,
            "salesperson_id": salesperson,
            "channel": channel,
            "quantity": quantity,
            "unit_price": round(unit_price, 2),
            "revenue": revenue,
            "profit_margin": profit_margin,
            "profit": round(revenue * profit_margin, 2),
        })

    df = pd.DataFrame(rows)

    # Inject anomalies: a few massive orders + a few near-zero revenue rows
    anomaly_idx = rng.choice(df.index, size=40, replace=False)
    df.loc[anomaly_idx[:20], "revenue"] = rng.uniform(15000, 50000, 20).round(2)
    df.loc[anomaly_idx[20:], "quantity"] = rng.integers(200, 500, 20)

    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2.  E-Commerce Orders
# ---------------------------------------------------------------------------

def generate_ecommerce_data(n: int = 8000) -> pd.DataFrame:
    cities = [
        ("New York", "USA"), ("Los Angeles", "USA"), ("Chicago", "USA"),
        ("Houston", "USA"), ("Phoenix", "USA"), ("London", "UK"),
        ("Manchester", "UK"), ("Toronto", "Canada"), ("Vancouver", "Canada"),
        ("Sydney", "Australia"),
    ]
    payment_methods = ["Credit Card", "PayPal", "Debit Card", "Bank Transfer", "Crypto"]
    categories = ["Electronics", "Fashion", "Books", "Home", "Beauty", "Toys"]
    return_reasons = ["Defective", "Wrong size", "Changed mind", "Not as described", ""]

    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2024-12-31")
    dates = pd.date_range(start, end, freq="D")

    rows = []
    for i in range(n):
        order_date = pd.Timestamp(rng.choice(dates))
        city, country = cities[rng.integers(len(cities))]
        category = rng.choice(categories)
        payment = rng.choice(payment_methods)

        # Electronics have higher return rate
        return_prob = 0.18 if category == "Electronics" else 0.06
        is_return = rng.random() < return_prob

        unit_price = float(rng.uniform(10, 800))
        qty = int(rng.integers(1, 6))
        discount = round(float(rng.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20])), 2)
        revenue = round(unit_price * qty * (1 - discount), 2)

        # Declining AOV trend in late 2024
        if order_date >= pd.Timestamp("2024-07-01"):
            revenue = round(revenue * 0.85, 2)

        rows.append({
            "order_id": f"ORD-{100000 + i}",
            "order_date": order_date.strftime("%Y-%m-%d"),
            "customer_id": f"CUST-{rng.integers(1, 2001):04d}",
            "category": category,
            "unit_price": round(unit_price, 2),
            "quantity": qty,
            "discount_pct": discount,
            "total_revenue": revenue,
            "is_return": int(is_return),
            "return_reason": rng.choice(return_reasons) if is_return else "",
            "payment_method": payment,
            "city": city,
            "country": country,
        })

    df = pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)

    # Anomaly: a handful of extremely high-value orders
    big_idx = rng.choice(df.index, size=25, replace=False)
    df.loc[big_idx, "total_revenue"] = rng.uniform(5000, 20000, 25).round(2)

    return df


# ---------------------------------------------------------------------------
# 3.  HR Analytics
# ---------------------------------------------------------------------------

def generate_hr_analytics(n: int = 1500) -> pd.DataFrame:
    departments = ["Sales", "Engineering", "Marketing", "HR", "Finance", "Operations"]
    job_roles = {
        "Sales": ["Sales Rep", "Account Manager", "Sales Manager"],
        "Engineering": ["Software Engineer", "Data Engineer", "DevOps", "QA Engineer"],
        "Marketing": ["Marketing Analyst", "Content Strategist", "SEO Specialist"],
        "HR": ["HR Coordinator", "Recruiter", "HR Manager"],
        "Finance": ["Financial Analyst", "Accountant", "CFO"],
        "Operations": ["Operations Manager", "Logistics Coordinator", "Supply Chain Analyst"],
    }
    education_levels = ["High School", "Bachelor's", "Master's", "PhD"]

    rows = []
    for i in range(n):
        dept = rng.choice(departments)
        role = rng.choice(job_roles[dept])
        education = rng.choice(education_levels, p=[0.05, 0.55, 0.35, 0.05])
        age = int(rng.integers(22, 60))
        years_at_co = int(rng.integers(0, min(age - 21, 20) + 1))

        base_salary = {
            "Sales": 55000, "Engineering": 95000, "Marketing": 65000,
            "HR": 50000, "Finance": 75000, "Operations": 60000,
        }[dept]
        salary = int(base_salary * rng.uniform(0.7, 1.6) + years_at_co * 1500)

        # Sales has higher attrition
        attrition_prob = 0.25 if dept == "Sales" else 0.08
        attrition_prob += 0.15 if salary < 45000 else 0.0
        attrition = int(rng.random() < attrition_prob)

        overtime = int(rng.random() < (0.4 if dept in ("Sales", "Engineering") else 0.2))
        performance = int(rng.integers(1, 6))
        satisfaction = round(float(rng.uniform(1.0, 5.0)), 1)
        wlb = int(rng.integers(1, 6))
        last_promo = int(rng.integers(0, years_at_co + 1))

        rows.append({
            "employee_id": f"EMP-{1000 + i}",
            "age": age,
            "gender": rng.choice(["Male", "Female", "Non-binary"], p=[0.52, 0.46, 0.02]),
            "department": dept,
            "job_role": role,
            "education": education,
            "years_at_company": years_at_co,
            "monthly_salary": salary,
            "performance_rating": performance,
            "satisfaction_score": satisfaction,
            "work_life_balance": wlb,
            "overtime": overtime,
            "attrition": attrition,
            "last_promotion_years_ago": last_promo,
        })

    df = pd.DataFrame(rows)

    # Salary anomalies in IT/Engineering
    eng_idx = df[df["department"] == "Engineering"].index
    anomaly_eng = rng.choice(eng_idx, size=min(15, len(eng_idx)), replace=False)
    df.loc[anomaly_eng, "monthly_salary"] = rng.integers(200000, 500000, len(anomaly_eng))

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating sample datasets...")

    sales = generate_sales_data(5000)
    sales.to_csv(OUTPUT_DIR / "sales_data.csv", index=False)
    print(f"  sales_data.csv       — {len(sales):,} rows")

    ecom = generate_ecommerce_data(8000)
    ecom.to_csv(OUTPUT_DIR / "ecommerce.csv", index=False)
    print(f"  ecommerce.csv        — {len(ecom):,} rows")

    hr = generate_hr_analytics(1500)
    hr.to_csv(OUTPUT_DIR / "hr_analytics.csv", index=False)
    print(f"  hr_analytics.csv     — {len(hr):,} rows")

    print(f"\nAll files saved to: {OUTPUT_DIR.resolve()}")
