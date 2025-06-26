# LangGraph Coding Agent

A coding agent built with LangGraph and LangChain that solves programming problems by:

1. Generating solution code in a Python class structure
2. Creating tests to validate the solution
3. Iterating on the code based on test results until the solution is correct or until MAX_ITERATIONS is reached

The agent is configurable, handles complex paths/spaces, and can solve advanced problems like CSV data analysis.

## Features

- **Configurable via JSON file**: All settings in `config.json`
- **Additional instructions**: Custom guidelines in `instructions.md` 
- **Memory checkpointing**: Supports in-memory state persistence
- **Robust file handling**: Works with paths containing spaces
- **CSV data processing**: Built-in support for data analysis tasks
- **Clean error handling**: Graceful recovery from failures
- **Max iterations limit**: Prevents infinite loops

## Installation

```bash
pip install -r requirements.txt
```

If you want to use AWS Bedrock models, make sure to configure your AWS credentials properly:

```bash
aws configure
```

## Usage

### Basic Usage

Run the agent with a problem description:

```bash
python agent.py "Write a function that calculates the Fibonacci sequence"
```

Provide expected output (optional):

```bash
python agent.py "Write a sorting algorithm that sorts an array in O(n log n) time" --output "Input: [5, 3, 8, 1, 2], Output: [1, 2, 3, 5, 8]"
```

### Running Examples

The repository includes example problems in the `Examples` directory:

```bash
cd Examples
python example1.py  # Simple factorial calculator
python example2.py  # CSV data processing example
```

### Command Line Options

- `--provider`: LLM provider to use (`openai` or `bedrock`, default from config)
- `--model`: Model name to use (default from config)
- `--temperature`: Temperature for generation (default from config)
- `--max-iterations`: Maximum improvement iterations (default from config)
- `--config`: Path to a custom configuration file

### Example with AWS Bedrock:

```bash
python agent.py "Write a function to check if a string is a palindrome" --provider bedrock --model "anthropic.claude-3-sonnet-20240229-v1:0"
```

### Example with Custom Settings:

```bash
python agent.py "Create a binary search tree implementation" --max-iterations 12 --temperature 0.2
```

## How It Works

1. **Problem Analysis**: The agent analyzes the problem and plans a solution approach
2. **Code Generation**: Code is generated in a Python class structure
3. **Test Creation**: Tests are created to validate the solution
4. **Iterative Improvement**: If tests fail, the agent improves the code and runs tests again
5. **Solution Finalization**: The agent provides the finalized solution when all tests pass

### Memory Management

The agent uses LangGraph for managing its state during execution:

- **In-Memory Checkpointing**: State can be maintained in memory (default)
- **Configurable Limits**: Control recursion depth and maximum iterations
- **Error Recovery**: Graceful handling of unexpected errors during execution
- **Memory Optimization**: Careful handling of large files and data structures

## Instructions File

The agent uses additional instructions from `instructions.md` to guide its problem-solving. The file is organized in sections starting with `##` headers:

```markdown
## Data Processing Guidelines
- When processing datasets, prefer using pandas for data manipulation
- Handle missing values explicitly (fillna, dropna)
- Use vectorized operations instead of loops

## Testing Guidelines
- Write tests that cover both common cases and edge cases
- Test each method independently with clear assertions
- Verify error handling for invalid inputs
```

These instructions are included in the system prompt and influence how the agent approaches problems. You can add, remove, or modify sections to customize the agent's behavior.

## Output

The agent will generate:
- `solution.py`: The solution code
- `test_solution.py`: Tests to validate the solution

## Troubleshooting

### Memory and Checkpointing Issues

- **Checkpointing Errors**: By default, checkpointing is disabled in `config.json` (`"use_checkpointing": false`). If you enable it, ensure your version of LangGraph supports it.
- **Recursion Errors**: If you see recursion limit errors, adjust the `recursion_limit` value in `config.json`.
- **Memory Usage**: For large problems, monitor memory usage as the agent keeps context in memory.

### Common Errors

- **API Key Not Found**: Ensure you've set the `OPENAI_API_KEY` environment variable.
- **Infinite Loops**: If the agent gets stuck in a loop, increase `max_iterations` and ensure code fixes are actually improving.
- **File Path Issues**: If you see errors about paths, ensure the problem doesn't involve invalid or inaccessible paths.

## Customization

### Configuration File

The agent behavior can be configured by editing `config.json`:

```json
{
  "max_iterations": 8,
  "llm_settings": {
    "default_model": "gpt-4o-mini",
    "default_provider": "openai",
    "default_temperature": 0.0
  },
  "agent_settings": {
    "verbose_logging": true,
    "cleanup_old_files": true
  },
  "graph_settings": {
    "recursion_limit": 50,
    "max_retries": 3,
    "use_checkpointing": false
  }
}
```

### Additional Instructions

You can add custom guidelines by editing `instructions.md`. The agent will incorporate these guidelines into its reasoning. For example:

```markdown
## Data Processing Guidelines
- When processing datasets, prefer using pandas for data manipulation
- Handle missing values explicitly (fillna, dropna)
- Use vectorized operations instead of loops

## Code Style Guidelines
- Follow PEP 8 conventions
- Write descriptive variable and function names
- Include type hints for function parameters and return values
```

### Other Customizations

- Adjust tool implementations in `agent.py` to extend the agent's capabilities
- Modify the system prompt in the `get_system_prompt()` function
- Change state handling in the agent graph nodes
