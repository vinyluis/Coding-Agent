"""
Example usage of the coding agent to solve a data processing problem:
Process sales data from CSV input and output metrics to a new CSV file.
"""

import os
import sys
import subprocess
import csv
import random
from datetime import datetime, timedelta

# Constants for CSV generation
PRODUCT_CATEGORIES = ["Electronics", "Clothing", "Food", "Books", "Home"]
REGIONS = ["North", "South", "East", "West"]
NUM_ROWS = 15  # Exceeding the minimum 10 rows requirement

def generate_sample_data(input_file="sales_data.csv"):
    """Generate sample sales data CSV."""
    print(f"Generating sample sales data CSV: {input_file}")
    
    # Generate random dates for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    with open(input_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Date', 'Product_Category', 'Region', 'Sales_Amount'])
        
        # Generate random sales data
        for _ in range(NUM_ROWS):
            # Random date within the last 30 days
            days_to_subtract = random.randint(0, 30)
            sale_date = (end_date - timedelta(days=days_to_subtract)).strftime('%Y-%m-%d')
            
            category = random.choice(PRODUCT_CATEGORIES)
            region = random.choice(REGIONS)
            sales_amount = round(random.uniform(100, 1000), 2)  # Sales between $100 and $1000
            
            writer.writerow([sale_date, category, region, sales_amount])
    
    print(f"Created sample data file with {NUM_ROWS} sales records")
    return input_file

def main():
    """Run the coding agent with a CSV data processing problem."""
    # Generate the sample data file
    input_file = generate_sample_data()
    output_file = "sales_metrics.csv"
    
    # Define problem for the coding agent
    problem = f"""
    Create a SalesAnalyzer class that processes sales data from a CSV file and generates metrics.
    
    Input: A CSV file named '{input_file}' with columns: 'Date', 'Product_Category', 'Region', 'Sales_Amount'.
    
    Output: A CSV file named '{output_file}' with the following metrics:
    1. Total sales by product category
    2. Total sales by region
    3. Average sales amount overall
    4. Highest selling product category
    5. Highest selling region
    6. Date with highest sales
    
    The SalesAnalyzer class should have methods to:
    - Read the input CSV file
    - Calculate each of the required metrics
    - Write the results to the output CSV file
    - Provide a summary method that returns all metrics as a dictionary
    """
    
    try:
        # Path to the agent.py file (one directory up from Examples)
        agent_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'agent.py')
        
        print("\nRunning coding agent...")
        cmd = [sys.executable, agent_path, problem]
        subprocess.run(cmd)
        
        # Check if solution exists and display results
        solution_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'solution.py')
        output_metrics_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_file)
        
        if os.path.exists(output_metrics_file):
            print("\nGenerated Metrics:")
            with open(output_metrics_file, 'r') as f:
                print(f.read())
        
    except Exception as e:
        print(f"Error running the agent: {str(e)}")

if __name__ == "__main__":
    main()
