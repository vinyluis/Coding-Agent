# Additional Instructions for the Coding Agent

These instructions will be added to the agent's prompts to improve its problem-solving capabilities.

## Data Processing Guidelines

- When processing datasets, prefer using pandas for data manipulation and analysis
- Use appropriate data types to optimize memory usage (e.g., categories for string columns with few unique values)
- Handle missing values explicitly (fillna, dropna) based on the specific requirements
- Use vectorized operations instead of loops when working with pandas DataFrames

## Machine Learning Guidelines

- For ML tasks, use scikit-learn's consistent API (fit/transform/predict pattern)
- Perform proper train/test splits to evaluate model performance
- Scale features when appropriate (especially for distance-based algorithms)
- Consider simple models first before trying more complex approaches
- Always include evaluation metrics appropriate for the problem (accuracy, precision/recall, RMSE, etc.)

## Code Style Guidelines

- Follow PEP 8 conventions for Python code style
- Write descriptive variable and function names that clearly indicate their purpose
- Include type hints for function parameters and return values
- Add meaningful docstrings for all classes and functions
- Handle edge cases explicitly (empty inputs, invalid values, etc.)

## Testing Guidelines

- Write tests that cover both common cases and edge cases
- Test each method independently with clear assertions
- Use appropriate test fixtures to set up test environments
- Verify error handling for invalid inputs

## File Processing Guidelines

- Use context managers (with statements) when working with files
- Handle paths in a platform-independent manner using os.path or pathlib
- Support both absolute and relative file paths
- Provide clear error messages for file-related issues

## Performance Guidelines

- Consider memory usage for large datasets
- Use appropriate data structures for the operations needed
- Optimize algorithms for time complexity when performance is critical
- Use generator expressions for memory efficiency when processing large sequences
