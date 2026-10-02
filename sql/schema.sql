-- Workforce360 PostgreSQL Schema
-- Base table for the IBM HR Attrition dataset.
-- Synthetic / extended fields are included as nullable columns for future phases.

DROP TABLE IF EXISTS employees;

CREATE TABLE employees (
    employee_id TEXT PRIMARY KEY,
    source_employee_number INTEGER UNIQUE,

    age INTEGER CHECK (age >= 0),
    gender TEXT,
    marital_status TEXT,
    education INTEGER CHECK (education BETWEEN 1 AND 5),
    education_field TEXT,

    department TEXT,
    job_role TEXT,
    job_level INTEGER CHECK (job_level BETWEEN 1 AND 5),
    employment_type TEXT,
    location TEXT,

    attrition BOOLEAN NOT NULL,
    business_travel TEXT,
    overtime BOOLEAN,
    distance_from_home INTEGER CHECK (distance_from_home >= 0),

    job_satisfaction INTEGER CHECK (job_satisfaction BETWEEN 1 AND 4),
    environment_satisfaction INTEGER CHECK (environment_satisfaction BETWEEN 1 AND 4),
    relationship_satisfaction INTEGER CHECK (relationship_satisfaction BETWEEN 1 AND 4),
    job_involvement INTEGER CHECK (job_involvement BETWEEN 1 AND 4),
    work_life_balance INTEGER CHECK (work_life_balance BETWEEN 1 AND 4),
    performance_rating INTEGER CHECK (performance_rating BETWEEN 1 AND 4),

    monthly_income NUMERIC(12, 2) CHECK (monthly_income >= 0),
    daily_rate NUMERIC(12, 2) CHECK (daily_rate >= 0),
    hourly_rate NUMERIC(12, 2) CHECK (hourly_rate >= 0),
    monthly_rate NUMERIC(12, 2) CHECK (monthly_rate >= 0),
    percent_salary_hike NUMERIC(5, 2) CHECK (percent_salary_hike >= 0),
    stock_option_level INTEGER CHECK (stock_option_level >= 0),

    total_working_years INTEGER CHECK (total_working_years >= 0),
    years_at_company INTEGER CHECK (years_at_company >= 0),
    years_in_current_role INTEGER CHECK (years_in_current_role >= 0),
    years_since_last_promotion INTEGER CHECK (years_since_last_promotion >= 0),
    years_with_curr_manager INTEGER CHECK (years_with_curr_manager >= 0),
    num_companies_worked INTEGER CHECK (num_companies_worked >= 0),
    training_times_last_year INTEGER CHECK (training_times_last_year >= 0),

    employee_count INTEGER,
    standard_hours INTEGER,
    over_18 TEXT,

    absenteeism INTEGER CHECK (absenteeism >= 0),
    promotion_history INTEGER CHECK (promotion_history >= 0),
    remote_work BOOLEAN,
    manager_rating NUMERIC(4, 2),
    engagement_score NUMERIC(5, 2),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_employees_department ON employees(department);
CREATE INDEX idx_employees_job_role ON employees(job_role);
CREATE INDEX idx_employees_attrition ON employees(attrition);
CREATE INDEX idx_employees_overtime ON employees(overtime);