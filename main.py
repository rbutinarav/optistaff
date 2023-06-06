import pandas as pd
import pulp


##DATASET AND REQUIREMENTS

# Job requirements
jobs = pd.DataFrame({
    'Job': ['Head Chef', 'Sous Chef', 'Line Cook', 'Senior Waitstaff', 'Junior Waitstaff', 'Cleaning Staff'],
    'English': [80, 70, 60, 70, 60, 50],
    'Italian': [80, 70, 60, 70, 60, 50],
    'Number': [1, 2, 4, 3, 3, 2]
})

# Candidates
candidates = pd.DataFrame({
    'Name': ['Tony Alfredo', 'Maria Russo', 'Vincent Garibaldi', 'Jennifer Davis', 'Bruce Knight', 
             'Emma Stone', 'James Bond', 'Olivia Munn', 'Tom Cruise', 'Nicole Kidman', 
             'Hugh Jackman', 'Scarlett Johansson', 'Charlize Theron', 'Angelina Jolie', 'Brad Pitt', 
             'Leonardo DiCaprio', 'Julia Roberts', 'Meryl Streep', 'Tom Hanks', 'Sandra Bullock', 
             'Daniel Craig', 'Natalie Portman', 'Kate Winslet', 'Robert Downey Jr', 'Johnny Depp', 
             'Keira Knightley', 'Emma Watson', 'Morgan Freeman', 'Matt Damon', 'Cate Blanchett'], 
    'Salary': [80000, 60000, 40000, 50000, 35000, 70000, 65000, 62000, 78000, 67000, 
               69000, 72000, 55000, 53000, 75000, 54000, 70000, 69000, 72000, 68000, 
               65000, 57000, 59000, 78000, 65000, 52000, 54000, 76000, 73000, 75000],
    'Role': ['Head Chef', 'Sous Chef', 'Line Cook', 'Senior Waitstaff', 'Junior Waitstaff', 
             'Sous Chef', 'Line Cook', 'Junior Waitstaff', 'Senior Waitstaff', 'Cleaning Staff', 
             'Head Chef', 'Sous Chef', 'Line Cook', 'Senior Waitstaff', 'Junior Waitstaff', 
             'Head Chef', 'Sous Chef', 'Line Cook', 'Senior Waitstaff', 'Junior Waitstaff', 
             'Cleaning Staff', 'Head Chef', 'Sous Chef', 'Line Cook', 'Senior Waitstaff', 
             'Junior Waitstaff', 'Cleaning Staff', 'Head Chef', 'Sous Chef', 'Line Cook'],
    'English': [100, 100, 60, 80, 70, 90, 75, 80, 85, 65, 80, 95, 70, 85, 80, 95, 70, 85, 
                80, 90, 85, 75, 80, 95, 80, 70, 85, 95, 80, 85],
    'Italian': [100, 100, 100, 90, 60, 85, 80, 70, 75, 60, 85, 90, 70, 80, 60, 80, 70, 85, 
                60, 85, 70, 80, 70, 90, 60, 70, 85, 80, 85, 70]
})


# Creating one hot encoded roles for candidates
for role in jobs['Job']:
    candidates[role] = (candidates['Role'] == role).astype(int)

# Create the 'prob' variable to contain the problem data
prob = pulp.LpProblem("Staffing Problem", pulp.LpMinimize)

# Create variables
x = pulp.LpVariable.dicts("CandidateAssigned", (candidates.index, jobs.index), cat='Binary')


##OBJECTIVE FUNCTION AND CONSTRAINTS


# Objective function
prob += pulp.lpSum([x[i][j] * candidates.loc[i, 'Salary'] for i in candidates.index for j in jobs.index])

# Constraints

# Each candidate can be assigned to at most one job
for i in candidates.index:
    prob += pulp.lpSum([x[i][j] for j in jobs.index]) <= 1

# Each job must be filled by the required number of candidates
for j in jobs.index:
    prob += pulp.lpSum([x[i][j] * candidates.loc[i, jobs.loc[j, 'Job']] for i in candidates.index]) == jobs.loc[j, 'Number']

# The total English proficiency of all candidates assigned must be at least the required total
prob += pulp.lpSum([x[i][j] * candidates.loc[i, 'English'] for i in candidates.index for j in jobs.index]) >= jobs['English'].sum()

# The total Italian proficiency of all candidates assigned must be at least the required total
prob += pulp.lpSum([x[i][j] * candidates.loc[i, 'Italian'] for i in candidates.index for j in jobs.index]) >= jobs['Italian'].sum()


##SOLVER AND RESULTS

# Solve the problem
prob.solve()

# Print the results
print("Status:", pulp.LpStatus[prob.status])

for i in candidates.index:
    for j in jobs.index:
        if x[i][j].varValue > 0:
            print(f"Candidate {candidates.loc[i, 'Name']} is assigned to job {jobs.loc[j, 'Job']}")
