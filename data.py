import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_gym_data(n_members=200, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    today = datetime.today()
    
    names = [
        "James Miller", "Maria Garcia", "Chris Johnson", "Ashley Williams",
        "Mike Brown", "Sarah Davis", "Kevin Wilson", "Jessica Moore",
        "Daniel Taylor", "Amanda Anderson", "Ryan Thomas", "Stephanie Jackson",
        "Justin White", "Brittany Harris", "Brandon Martin", "Samantha Thompson",
        "Tyler Garcia", "Lauren Martinez", "Nathan Robinson", "Megan Clark"
    ]
    
    martial_arts = ["BJJ", "MMA", "Muay Thai", "Boxing", "Wrestling"]
    
    members = []
    for i in range(n_members):
        join_date = today - timedelta(days=np.random.randint(30, 730))
        membership_months = (today - join_date).days / 30
        
        # Churn risk factors
        base_attendance = np.random.randint(1, 5)  # sessions per week
        attendance_drop = np.random.choice([True, False], p=[0.3, 0.7])
        days_since_last = np.random.randint(1, 60)
        failed_payments = np.random.randint(0, 3)
        
        # Calculate churn probability
        churn_score = 0
        if days_since_last > 21:
            churn_score += 40
        if days_since_last > 14:
            churn_score += 20
        if attendance_drop:
            churn_score += 25
        if failed_payments > 0:
            churn_score += failed_payments * 15
        if membership_months < 3:
            churn_score += 10
        churn_score = min(churn_score, 95)
        
        churned = churn_score > 60 and np.random.random() < 0.6
        
        members.append({
            "member_id": f"MBR{1000+i}",
            "name": random.choice(names) + f" {i}",
            "martial_art": random.choice(martial_arts),
            "join_date": join_date.strftime("%Y-%m-%d"),
            "membership_months": round(membership_months, 1),
            "avg_sessions_per_week": base_attendance,
            "days_since_last_visit": days_since_last,
            "attendance_drop": int(attendance_drop),
            "failed_payments": failed_payments,
            "monthly_fee": random.choice([150, 175, 200]),
            "churned": int(churned),
            "churn_score": churn_score
        })
    
    return pd.DataFrame(members)

if __name__ == "__main__":
    df = generate_gym_data()
    print(df.head())
    print(f"\nTotal members: {len(df)}")
    print(f"Churned: {df['churned'].sum()}")
    print(f"At risk: {len(df[df['churn_score'] > 40])}")