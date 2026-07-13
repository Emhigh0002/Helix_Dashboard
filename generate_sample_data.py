import os
import csv
import random
from datetime import datetime, timedelta

def generate_data(num_rows=200):
    columns = [
        "StudentID", "Gender", "Age", "Major", "StudyHoursPerWeek", 
        "AttendanceRate", "MidtermScore", "FinalScore", "Passed", "EnrollmentDate"
    ]
    
    majors = ["Computer Science", "Data Science", "Business Analytics", "Mathematics", "Physics"]
    genders = ["Male", "Female"]
    
    rows = []
    base_date = datetime(2025, 9, 1)
    
    for i in range(1, num_rows + 1):
        std_id = 1000 + i
        gender = random.choice(genders)
        
        # Inject some missing values occasionally
        age = random.randint(18, 26) if random.random() > 0.05 else None
        major = random.choice(majors) if random.random() > 0.05 else None
        
        study_hours = round(random.uniform(5, 30), 1) if random.random() > 0.08 else None
        attendance = round(random.uniform(60, 100), 2)
        
        # Midterm and Final score with correlation
        base_performance = random.uniform(40, 95)
        midterm = round(base_performance + random.uniform(-5, 5), 1)
        final = round(base_performance * 1.1 + random.uniform(-8, 8), 1)
        
        # Clip scores
        midterm = max(0, min(100, midterm))
        final = max(0, min(100, final))
        
        passed = "Yes" if final >= 60 else "No"
        
        # Enroll date
        enroll_date = (base_date + timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        
        # Inject duplicate rows
        rows.append([std_id, gender, age, major, study_hours, attendance, midterm, final, passed, enroll_date])
        
    # Inject 3 duplicate rows
    for _ in range(3):
        rows.append(random.choice(rows).copy())
        
    return columns, rows

def main():
    cols, data = generate_data()
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_performance_sample.csv")
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(data)
    print(f"Generated sample dataset at {file_path}")

if __name__ == "__main__":
    main()
