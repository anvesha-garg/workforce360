# Workforce360 Data Dictionary

## Dataset

**Base dataset:** IBM HR Analytics Employee Attrition & Performance  
**Raw file path:** `data/raw/ibm_hr_attrition.csv`  
**Primary identifier:** `EmployeeNumber`  
**Prediction target:** `Attrition`

## Core Employee Attributes

| Column | Data Type | Description | Allowed Values / Notes |
|---|---|---|---|
| `EmployeeNumber` | Integer | Unique employee identifier in the source dataset | Expected to be unique |
| `Age` | Integer | Employee age in years | Positive integer |
| `Gender` | Text | Employee gender | `Male`, `Female` |
| `MaritalStatus` | Text | Employee marital-status category | `Single`, `Married`, `Divorced` |
| `Education` | Integer | Education level | 1 = Below College; 2 = College; 3 = Bachelor; 4 = Master; 5 = Doctor |
| `EducationField` | Text | Field of education | Examples: Life Sciences, Medical, Marketing, Technical Degree |
| `Department` | Text | Employee department | Research & Development, Sales, Human Resources |
| `JobRole` | Text | Employee job role | Examples: Sales Executive, Research Scientist, Laboratory Technician |
| `JobLevel` | Integer | Job seniority level | Integer from 1 to 5 |

## Attrition and Work Factors

| Column | Data Type | Description | Allowed Values / Notes |
|---|---|---|---|
| `Attrition` | Text | Whether the employee left the organization | `Yes`, `No` |
| `OverTime` | Text | Whether the employee works overtime | `Yes`, `No` |
| `BusinessTravel` | Text | Business travel frequency | `Non-Travel`, `Travel_Rarely`, `Travel_Frequently` |
| `DistanceFromHome` | Integer | Distance from home to workplace | Non-negative integer |
| `WorkLifeBalance` | Integer | Work-life balance rating | 1 = Bad; 2 = Good; 3 = Better; 4 = Best |
| `JobSatisfaction` | Integer | Job satisfaction rating | 1 = Low; 2 = Medium; 3 = High; 4 = Very High |
| `EnvironmentSatisfaction` | Integer | Satisfaction with work environment | 1 = Low; 2 = Medium; 3 = High; 4 = Very High |
| `RelationshipSatisfaction` | Integer | Relationship satisfaction rating | 1 = Low; 2 = Medium; 3 = High; 4 = Very High |
| `JobInvolvement` | Integer | Job involvement rating | 1 = Low; 2 = Medium; 3 = High; 4 = Very High |
| `PerformanceRating` | Integer | Employee performance rating | Typically 3 = Excellent; 4 = Outstanding |

## Compensation and Benefits

| Column | Data Type | Description | Allowed Values / Notes |
|---|---|---|---|
| `MonthlyIncome` | Integer | Monthly employee income | Positive integer |
| `DailyRate` | Integer | Daily pay rate | Positive integer |
| `HourlyRate` | Integer | Hourly pay rate | Positive integer |
| `MonthlyRate` | Integer | Monthly pay rate field in source data | Positive integer |
| `PercentSalaryHike` | Integer | Percentage salary increase | Percentage integer |
| `StockOptionLevel` | Integer | Stock-option level | Non-negative integer |

## Experience and Career History

| Column | Data Type | Description | Allowed Values / Notes |
|---|---|---|---|
| `TotalWorkingYears` | Integer | Total years of work experience | Non-negative integer |
| `YearsAtCompany` | Integer | Years at the current company | Non-negative integer |
| `YearsInCurrentRole` | Integer | Years in current job role | Non-negative integer |
| `YearsSinceLastPromotion` | Integer | Years since last promotion | Non-negative integer |
| `YearsWithCurrManager` | Integer | Years with current manager | Non-negative integer |
| `NumCompaniesWorked` | Integer | Number of prior companies worked for | Non-negative integer |
| `TrainingTimesLastYear` | Integer | Training events completed in the last year | Non-negative integer |

## Dataset Constants / Potentially Low-Value Fields

| Column | Data Type | Description | Notes |
|---|---|---|---|
| `EmployeeCount` | Integer | Employee-count field | Often constant in this dataset |
| `Over18` | Text | Whether employee is over 18 | Often constant: `Y` |
| `StandardHours` | Integer | Standard working hours | Often constant in this dataset |

## Future Workforce360 Fields

The base IBM dataset does not include all planned Workforce360 fields. The following may be introduced later through a clearly labeled synthetic data extension:

| Future Field | Intended Use |
|---|---|
| `employee_id` | Standardized Workforce360 employee ID, e.g. `EMP1024` |
| `employment_type` | Full-time, part-time, contractor |
| `location` | Office/city/region |
| `absenteeism` | Absence days or rate |
| `promotion_history` | Promotion count or promotion indicator |
| `remote_work` | Remote/hybrid/on-site indicator |
| `manager_rating` | Manager evaluation score |
| `engagement_score` | Employee engagement score |
| `hire_date` | Needed for realistic monthly hiring and attrition forecasting |