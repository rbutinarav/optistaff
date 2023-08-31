import pandas as pd
import pulp
import streamlit as st

def load_data():
    jobs = pd.read_csv('data/jobs.csv')
    candidates = pd.read_csv('data/candidates.csv')

    return jobs, candidates

def create_problem(jobs, candidates):
    # Create the 'prob' variable to contain the problem data
    prob = pulp.LpProblem("Staffing Problem", pulp.LpMinimize)

    # Create variables
    x = pulp.LpVariable.dicts("CandidateAssigned", 
                              ((i, j) for i in candidates.index for j in jobs.index if candidates.loc[i, 'Role'] == jobs.loc[j, 'Job']), 
                              cat='Binary')

    # Create a slack variable for each job
    slack = pulp.LpVariable.dicts("Slack", jobs.index, lowBound=0)

    # Objective function
    prob += pulp.lpSum([x[(i, j)] * candidates.loc[i, 'Salary'] for (i, j) in x]) \
            + 10000 * pulp.lpSum(slack[j] for j in jobs.index)

    # Constraints

    # Each candidate can be assigned to at most one job
    for i in candidates.index:
        prob += pulp.lpSum([x[(i, j)] for j in jobs.index if (i, j) in x]) <= 1

    # Each job must be filled by the required number of candidates
    for j in jobs.index:
        prob += pulp.lpSum([x[(i, j)] for i in candidates.index if (i, j) in x]) == jobs.loc[j, 'Number']

    # The total English proficiency of the selected candidates must be at least the required proficiency for each job
    for j in jobs.index:
        prob += pulp.lpSum([x[(i, j)] * candidates.loc[i, 'English'] for i in candidates.index if (i, j) in x]) >= jobs.loc[j, 'English'] * jobs.loc[j, 'Number']

    # The total Italian proficiency of the selected candidates plus the slack variable must be at least the required proficiency for each job
    for j in jobs.index:
        prob += pulp.lpSum([x[(i, j)] * candidates.loc[i, 'Italian'] for i in candidates.index if (i, j) in x]) + slack[j] >= jobs.loc[j, 'Italian'] * jobs.loc[j, 'Number']

    return prob, x

def get_results(prob, x, jobs, candidates):
    # Solve the problem
    prob.solve()

    # Print the results
    status = pulp.LpStatus[prob.status]
    #st.write("Status:", status)

    #if status == "Optimal":
    #    st.write("Optimal Solution Found:")
    #else:
    #    st.write("No optimal solution found. The requirements could not be met with the available candidates. This is the best solution found:")
        
    # Create a DataFrame to store the assignments
    assignments = pd.DataFrame(columns=['Job', 'Candidate Name', 'English Proficiency', 'Italian Proficiency', 'Salary'])

    for (i, j) in x:
        if x[(i, j)].varValue > 0:
            assignments = pd.concat([assignments, pd.DataFrame([{'Job': jobs.loc[j, 'Job'], 
                                                                'Candidate Name': candidates.loc[i, 'Name'], 
                                                                'English Proficiency': candidates.loc[i, 'English'], 
                                                                'Italian Proficiency': candidates.loc[i, 'Italian'],
                                                                'Salary': candidates.loc[i, 'Salary']}],
                                                                columns=assignments.columns)],
                                        ignore_index=True)

    assignments = assignments.sort_values(by=['Job', 'Candidate Name'])

    return assignments, status

def display_assignements(assignments):
    st.write("**Assignments**")
    st.write(assignments)

def display_summary(assignments, jobs):
    # Calculate total and average cost, number of found resources, and average proficiencies
    summary = pd.DataFrame(columns=['Job', 'Required Number', 'Required English', 'Required Italian',
                                    'Found Resources', 'Average English', 'Average Italian', 'Total Cost', 'Average Cost',
                                    'Unfilled Positions', 'Difference English', 'Difference Italian'])

    for j in jobs.index:
        job = jobs.loc[j, 'Job']
        required_number = jobs.loc[j, 'Number']
        required_english = jobs.loc[j, 'English']
        required_italian = jobs.loc[j, 'Italian']
        found_resources = assignments[assignments['Job'] == job].shape[0]
        total_cost = assignments[assignments['Job'] == job]['Salary'].sum()
        average_english = round(assignments[assignments['Job'] == job]['English Proficiency'].mean())
        average_italian = round(assignments[assignments['Job'] == job]['Italian Proficiency'].mean())
        average_cost = round(assignments[assignments['Job'] == job]['Salary'].mean())
        unfilled_positions = found_resources - required_number
        difference_english = average_english - required_english
        difference_italian = average_italian - required_italian
        
        summary = pd.concat([summary, pd.DataFrame([{'Job': job, 
                                                    'Required Number': required_number,
                                                    'Required English': required_english,
                                                    'Required Italian': required_italian,
                                                    'Found Resources': found_resources,
                                                    'Average English': average_english,
                                                    'Average Italian': average_italian,
                                                    'Total Cost': total_cost,
                                                    'Average Cost': average_cost,
                                                    'Unfilled Positions': unfilled_positions,
                                                    'Difference English': difference_english,
                                                    'Difference Italian': difference_italian}],
                                                    columns=summary.columns)], 
                            ignore_index=True)

    # Append total row to summary
    summary.loc['Total'] = [
        'Total',
        summary['Required Number'].sum(),
        round(summary['Required English'].mean()),
        round(summary['Required Italian'].mean()),
        summary['Found Resources'].sum(),
        round(summary['Average English'].mean()),
        round(summary['Average Italian'].mean()),
        summary['Total Cost'].sum(),
        round(summary['Total Cost'].sum() / summary['Found Resources'].sum()),
        summary['Unfilled Positions'].sum(),
        round(summary['Average English'].mean()) - round(summary['Required English'].mean()),
        round(summary['Average Italian'].mean()) - round(summary['Required Italian'].mean())
    ]


    # Select columns to display
    summary_display = summary[['Job','Found Resources', 'Total Cost', 'Average Cost', 'Average English', 'Average Italian',
                               'Unfilled Positions', 'Difference English', 'Difference Italian']]

    st.write("Summary")
    st.write(summary_display)


def header():
    st.write("# Optistaff: Staffing Optimization")

    st.write("**Scenario**")
    st.write("This solution will solve the optimal staff allocation given a set of job positions to fulfill\
            specific requirements of skills, a list of candidates and a set of goals.")

    st.write("**Jobs**")
    st.write("For each Job, a target of English and Italian average proficiency is set for the\
            total number of positions to fill")

    st.write("**Candidates**")
    st.write("The company has a list of candidates that can be assigned to different roles. The\
            company already knows the proficiency of candidates in English and Italian languages and the salary they are asking.")

    st.write("**Goals (Optimization)**")
    st.write("The company wants to determine the assignment of candidates to roles that will minimize\
            the overall cost of hiring the employees, while still satisfying the required language\
            proficiencies for each role. If no solution can be found, the suboptimal solution will be displayed.")

def model_description():
    st.write("# Model")

    st.write("**Objective Function**")
    st.write("Minimize the total cost of hiring the employees.")

    st.write("**Constraints**")
    st.write("1. Each candidate can be assigned to at most one job.")
    st.write("2. Each job must be filled by the required number of candidates.")
    st.write("3. The total English proficiency of the selected candidates must be at least the required proficiency for each job.")
    st.write("4. The total Italian proficiency of the selected candidates plus a slack variable must be at least the required proficiency for each job. The slack variable allows the optimizer to select candidates that do not satisfy the Italian proficiency, but this will result in a higher total cost.")


def main():
    # Sidebar controls
    show_header = st.sidebar.checkbox("Show Header", True)
    show_input_data = st.sidebar.checkbox("Show Input Data", True)
    run_solver = st.sidebar.button("Run Solver")
    show_results = st.sidebar.checkbox("Show Results", True)

    # 1. describe the solution
    if show_header:
        header()

    # 2. load and display the input data
    if show_input_data:
        st.write("# Input Data")

        st.write("**Job Data**")
        jobs, candidates = load_data()

        st.write(jobs)

        st.write("**Candidate Data**")
        st.write(candidates)
    else:
        jobs, candidates = load_data()

    # 3. solve the problem
    if run_solver:
        st.write("# Solution")

        prob, x = create_problem(jobs, candidates)

        assignments, status = get_results(prob, x, jobs, candidates)
    else:
        assignments = None

    # 4. display the results
    if show_results and assignments is not None:

        display_assignements(assignments)
        display_summary(assignments, jobs)

# Execute the main function
if __name__ == "__main__":
    main()

