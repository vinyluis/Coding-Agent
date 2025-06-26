# Examples for LangGraph Coding Agent

This folder contains examples of how to use the LangGraph Coding Agent for solving different programming problems. These examples demonstrate how the agent can handle various types of programming tasks, from simple algorithmic problems to more complex data processing tasks.

## Example 1: Factorial Calculator

**File**: `example1.py`

This example demonstrates the coding agent creating a class that calculates factorials. The agent will:

1. Generate a `Factorial` class in Python that can calculate the factorial of a positive integer
2. Create comprehensive tests for the class
3. Execute the tests to validate the solution
4. Fix any errors found during testing

This is a good starting example to understand how the agent works with simpler algorithmic problems.

To run this example:
```bash
python example1.py
```

For more detailed output:
```bash
python example1.py --verbose
```

## Example 2: CSV Data Processor

**File**: `example2.py`

This example demonstrates the coding agent handling more complex data processing tasks with CSV files. The agent will:

1. Generate sample sales data in `sales_data.csv` (if it doesn't exist)
2. Create a `SalesAnalyzer` class that:
   - Reads the input sales data CSV
   - Calculates metrics like total sales by category, region, averages, etc.
   - Identifies top-performing categories and regions
   - Outputs the results to `sales_metrics.csv`
3. Develop comprehensive tests to validate the solution
4. Fix any errors that arise during testing

This example showcases how the agent can work with real-world data processing problems using pandas.

To run this example:
```bash
python example2.py
```

You can override the default configuration settings:
```bash
python example2.py --max-iterations 10 --model gpt-4o
```

## Notes

### Prerequisites

- Ensure all dependencies are installed: `pip install -r ../requirements.txt`
- Set your OpenAI API key: `export OPENAI_API_KEY=your_key_here` (Linux/Mac) or `set OPENAI_API_KEY=your_key_here` (Windows)
- For AWS Bedrock: Configure AWS credentials with `aws configure`

### How the Examples Work

The examples use the agent from the parent directory to solve specific programming problems. Each example:

1. Defines a specific programming problem
2. Calls the main function from `agent.py` with this problem
3. Generates a solution and tests in the current directory
4. Validates the solution by running the tests

### Additional Configuration

- All examples respect the settings in `../config.json`
- You can modify `../instructions.md` to change how the agent approaches problems
- Command line arguments will override config file settings
