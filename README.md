# LangGraph Coding Agent

A coding agent built with LangGraph and LangChain that solves programming problems by:

1. Generating solution code in a Python class structure
2. Creating tests to validate the solution
3. Iterating on the code based on test results until the solution is correct or until MAX_ITERATIONS is reached

## Installation

```bash
pip install -r requirements.txt
```

If you want to use AWS Bedrock models, make sure to configure your AWS credentials properly:

```bash
aws configure
```

## Usage

Run the agent with a problem description:

```bash
python agent.py "Write a function that calculates the Fibonacci sequence"
```

Provide expected output (optional):

```bash
python agent.py "Write a sorting algorithm that sorts an array in O(n log n) time" --output "Input: [5, 3, 8, 1, 2], Output: [1, 2, 3, 5, 8]"
```

### Options

- `--provider`: LLM provider to use (`openai` or `bedrock`, default: `openai`)
- `--model`: Model name to use (default: `gpt-4o` for OpenAI, `anthropic.claude-3-sonnet-20240229-v1:0` for Bedrock)
- `--temperature`: Temperature for generation (default: `0.0`)

Example using AWS Bedrock:

```bash
python agent.py "Write a function to check if a string is a palindrome" --provider bedrock --model "anthropic.claude-3-sonnet-20240229-v1:0"
```

## How It Works

1. **Problem Analysis**: The agent analyzes the problem and plans a solution approach
2. **Code Generation**: Code is generated in a Python class structure
3. **Test Creation**: Tests are created to validate the solution
4. **Iterative Improvement**: If tests fail, the agent improves the code and runs tests again
5. **Solution Finalization**: The agent provides the finalized solution when all tests pass

## Output

The agent will generate:
- `solution.py`: The solution code
- `test_solution.py`: Tests to validate the solution

## Customization

You can modify the agent behavior by editing the `agent.py` file:
- Adjust `MAX_ITERATIONS` to control how many improvement cycles the agent will attempt
- Update tool implementations to extend the agent's capabilities
- Change the system prompt to adjust the agent's instructions
