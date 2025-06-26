#!/usr/bin/env python3
"""
LangGraph Coding Agent that solves programming problems by:
1. Generating code in a class structure
2. Creating tests to validate the solution
3. Iterating on the code based on test results
"""

import os
import re
import json
import subprocess
import argparse
import os.path as ospath
import time
from typing import Dict, Any, TypedDict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
try:
  from langchain_aws import ChatBedrock
except ImportError:
  # Optional dependency, will be None if not installed
  ChatBedrock = None
from langchain_core.tools import ToolException, tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
# tempfile import removed - not needed for in-memory checkpointing

# Load configuration from file
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
DEFAULT_CONFIG = {
  "max_iterations": 8,
  "llm_settings": {
    "default_model": "gpt-4o-mini",
    "default_provider": "openai",
    "default_temperature": 0.0
  },
  "agent_settings": {
    "verbose_logging": True,
    "cleanup_old_files": True
  }
}  # Load config or use defaults if file doesn't exist
try:
  with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)
    print(f"✅ Loaded configuration from {CONFIG_FILE}")
except (FileNotFoundError, json.JSONDecodeError) as e:
  print(f"⚠️ Could not load config file: {e}")
  print(f"⚠️ Using default configuration")
  CONFIG = DEFAULT_CONFIG

# Check for instructions file
INSTRUCTIONS_FILE = os.path.join(os.path.dirname(__file__), "instructions.md")
if os.path.exists(INSTRUCTIONS_FILE):
  print(f"✅ Found instructions file at {INSTRUCTIONS_FILE}")
  try:
    with open(INSTRUCTIONS_FILE, "r", encoding="utf-8") as f:
      inst_content = f.read().splitlines()
      inst_sections = [line for line in inst_content if line.startswith('## ')]
      print(f"📋 Found {len(inst_sections)} instruction sections: {', '.join([s.strip('## ') for s in inst_sections])}")
  except Exception as e:
    print(f"⚠️ Error reading instructions file: {str(e)}")

# Configure maximum iterations from config
MAX_ITERATIONS = CONFIG.get("max_iterations", 8)


# Type definitions for the graph state
class State(TypedDict):
  messages: list
  problem_description: str
  expected_output: Optional[str]
  code_file: Optional[str]
  test_file: Optional[str]
  test_results: Optional[str]
  iteration_count: int
  next: Optional[str]

# Tools for the agent to interact with the environment

@tool
def read_file(file_path: str) -> str:
  """
  Read the contents of a file.
  
  Args:
    file_path: Path to the file to read
    
  Returns:
    The contents of the file as a string
  """
  try:
    # Normalize the path to handle spaces and special characters
    file_path = os.path.normpath(os.path.abspath(file_path))
    
    with open(file_path, "r", encoding='utf-8') as f:
      return f.read()
  except Exception as e:
    raise ToolException(f"Error reading file: {str(e)}")

@tool
def write_file(file_path: str, content: str) -> str:
  """
  Write content to a file.
  
  Args:
    file_path: Path to the file to write
    content: Content to write to the file
    
  Returns:
    Confirmation message
  """
  try:
    # Check if file_path is empty or None
    if not file_path:
      raise ValueError("File path cannot be empty")
    
    # Normalize the path to handle spaces and special characters
    file_path = os.path.normpath(os.path.abspath(file_path))
    
    # If file_path is just a filename with no directory, use the current directory
    if os.path.dirname(file_path) == '':
      file_path = os.path.normpath(os.path.join(os.getcwd(), file_path))
      print(f"Using absolute path: {file_path}")
    
    # Ensure directory exists
    dir_path = os.path.dirname(file_path)
    if dir_path:  # Only try to create directory if there is a directory component
      os.makedirs(dir_path, exist_ok=True)
      
    # Write the file
    with open(file_path, "w", encoding='utf-8') as f:
      f.write(content)
    
    return f"Successfully wrote to {file_path}"
  except Exception as e:
    raise ToolException(f"Error writing to file: {str(e)}")

@tool
def edit_file(file_path: str, old_content: str, new_content: str) -> str:
  """
  Edit a file by replacing old content with new content.
  
  Args:
    file_path: Path to the file to edit
    old_content: Content to replace
    new_content: Content to replace with
    
  Returns:
    Confirmation message
  """
  try:
    # Normalize the path to handle spaces and special characters
    file_path = os.path.normpath(os.path.abspath(file_path))
    
    if not os.path.exists(file_path):
      return f"Error: File {file_path} does not exist"
    
    with open(file_path, "r", encoding='utf-8') as f:
      content = f.read()
    
    if old_content not in content:
      return f"Error: Could not find the content to replace in {file_path}"
    
    updated_content = content.replace(old_content, new_content)
    
    with open(file_path, "w", encoding='utf-8') as f:
      f.write(updated_content)
    
    return f"Successfully edited {file_path}"
  except Exception as e:
    raise ToolException(f"Error editing file: {str(e)}")

@tool
def run_command(command: str) -> str:
  """
  Run a shell command.
  
  Args:
    command: Command to run
    
  Returns:
    Command output
  """
  try:
    result = subprocess.run(
      command,
      shell=True,
      check=True,
      text=True,
      capture_output=True
    )
    return result.stdout
  except subprocess.CalledProcessError as e:
    return f"Command failed with exit code {e.returncode}\nStdout: {e.stdout}\nStderr: {e.stderr}"
  except Exception as e:
    raise ToolException(f"Error running command: {str(e)}")

@tool
def search_text(file_path: str, search_term: str) -> str:
  """
  Search for text in a file.
  
  Args:
    file_path: Path to the file to search in
    search_term: Text to search for
    
  Returns:
    Lines containing the search term
  """
  try:
    # Normalize the path to handle spaces and special characters
    file_path = os.path.normpath(os.path.abspath(file_path))
    
    if not os.path.exists(file_path):
      return f"Error: File {file_path} does not exist"
    
    with open(file_path, "r", encoding='utf-8') as f:
      content = f.readlines()
    
    matches = [line.strip() for line in content if search_term in line]
    
    if matches:
      return f"Found {len(matches)} matches:\n" + "\n".join(matches)
    else:
      return f"No matches found for '{search_term}' in {file_path}"
  except Exception as e:
    raise ToolException(f"Error searching text: {str(e)}")

@tool
def run_tests(test_file: str) -> str:
  """
  Run Python tests.
  
  Args:
    test_file: Path to the test file
    
  Returns:
    Test results
  """
  try:
    # Normalize the path to handle spaces and special characters
    test_file = os.path.normpath(os.path.abspath(test_file))
    
    # Quote the path to handle spaces in subprocess
    quoted_path = f'"{test_file}"'
    
    result = subprocess.run(
      f"python -m pytest {quoted_path} -v",
      shell=True,
      text=True,
      capture_output=True
    )
    
    # Return both stdout and stderr to provide full test information
    output = result.stdout
    if result.stderr:
      output += "\n" + result.stderr
      
    return output
  except Exception as e:
    raise ToolException(f"Error running tests: {str(e)}")

@tool
def list_directory(directory_path: str = "./") -> str:
  """
  List files in a directory.
  
  Args:
    directory_path: Path to the directory to list
    
  Returns:
    List of files in the directory
  """
  try:
    # Normalize the path to handle spaces and special characters
    directory_path = os.path.normpath(os.path.abspath(directory_path))
    
    files = os.listdir(directory_path)
    return "\n".join(files)
  except Exception as e:
    raise ToolException(f"Error listing directory: {str(e)}")

@tool
def get_current_directory() -> str:
  """
  Get the current working directory.
  
  Returns:
    The absolute path of the current working directory
  """
  try:
    # Return normalized path to handle any inconsistencies
    return os.path.normpath(os.getcwd())
  except Exception as e:
    raise ToolException(f"Error getting current directory: {str(e)}")

def load_instructions() -> str:
  """
  Load additional instructions from instructions.md file if it exists.
  
  Returns:
    String containing instructions or empty string if file doesn't exist
  """
  try:
    # Use the global INSTRUCTIONS_FILE constant
    if os.path.exists(INSTRUCTIONS_FILE):
      with open(INSTRUCTIONS_FILE, "r", encoding="utf-8") as f:
        content = f.read()
        print(f"✅ Loaded {len(content)} characters of instructions from {INSTRUCTIONS_FILE}")
        return content
    else:
      print("⚠️ No instructions.md file found. Using default instructions only.")
      return ""
  except Exception as e:
    print(f"⚠️ Error loading instructions: {str(e)}")
    return ""


@tool
def change_directory(directory_path: str) -> str:
  """
  Change the current working directory.
  
  Args:
    directory_path: Path to the directory to change to
    
  Returns:
    Confirmation message
  """
  try:
    # Normalize the path to handle spaces and special characters
    directory_path = os.path.normpath(os.path.abspath(directory_path))
    
    os.chdir(directory_path)
    return f"Changed working directory to {os.path.normpath(os.getcwd())}"
  except Exception as e:
    raise ToolException(f"Error changing directory: {str(e)}")

@tool
def ensure_directory(directory_path: str) -> str:
  """
  Ensure a directory exists, creating it if necessary.
  
  Args:
    directory_path: Path to the directory to ensure exists
    
  Returns:
    Confirmation message
  """
  try:
    # Normalize the path to handle spaces and special characters
    directory_path = os.path.normpath(os.path.abspath(directory_path))
    
    os.makedirs(directory_path, exist_ok=True)
    return f"Ensured directory exists: {directory_path}"
  except Exception as e:
    raise ToolException(f"Error ensuring directory exists: {str(e)}")

# Define agent nodes

def create_llm(temperature: float = 0.0, model_name: str = "gpt-4o-mini", provider: str = "openai"):
  """Create a language model.
  
  Args:
    temperature: The temperature for generation
    model_name: The name of the model to use
    provider: The provider of the model (openai or bedrock)
    
  Returns:
    A language model instance
  """
  try:
    if provider.lower() == "bedrock" and ChatBedrock is not None:
      return ChatBedrock(
        model=model_name,
        temperature=temperature
      )
    else:
      # Default to OpenAI
      return ChatOpenAI(
        temperature=temperature,
        model=model_name
      )
  except Exception as e:
    print(f"❌ Error initializing the language model: {str(e)}")
    print(f"Provider: {provider}, Model: {model_name}")
    raise

def get_system_prompt():
  # Load additional instructions if available
  additional_instructions = load_instructions()
  
  base_prompt = """You are a professional coding agent that solves programming problems.
  
You are tasked with solving a problem by:
1. Understanding the requirements
2. Creating a solution in a Python class structure
3. Creating tests for the solution
4. Improving the code based on test results

You have access to tools that allow you to read/write files, run commands, search text, run tests, and more.
Be thorough in your analysis and methodical in your approach.
"""

  # Add additional instructions if they exist
  if additional_instructions:
    print("🔍 Including additional instructions in system prompt")
    full_prompt = f"{base_prompt}\n\n## ADDITIONAL INSTRUCTIONS\n{additional_instructions}"
    return full_prompt
  else:
    print("⚠️ No additional instructions found for system prompt")
    return base_prompt

def initialize_state(state: State) -> State:
  """Initialize the agent state with the problem description."""
  messages = state["messages"]
  
  # Extract problem description and expected output
  problem_description = state.get("problem_description", "")
  expected_output = state.get("expected_output", None)
  
  # Get system prompt with instructions included
  system_prompt = get_system_prompt()
  
  # Check if additional instructions are included in the system prompt
  if "ADDITIONAL INSTRUCTIONS" in system_prompt:
    print("🎯 Successfully included instructions in system message")
  else:
    print("⚠️ No additional instructions included in system message")
  
  # Add system message with the agent's instructions
  messages.append(SystemMessage(content=system_prompt))
  
  # Add problem description as a human message
  human_message = f"Problem to solve: {problem_description}"
  if expected_output:
    human_message += f"\n\nExpected output: {expected_output}"
  
  messages.append(HumanMessage(content=human_message))
  
  # Initialize tracking variables
  return {
    **state,
    "messages": messages,
    "iteration_count": 0,
    "code_file": None,
    "test_file": None,
    "test_results": None,
    "next": "think"
  }

def think(state: State, llm) -> State:
  """Think about the problem and plan the solution approach."""
  print("🤔 Agent is thinking about the problem...")
  messages = state["messages"]
  
  # Get additional instructions for references
  additional_instructions = load_instructions()
  thinking_prompt = """
      Think through this problem step by step:
      1. What is the core problem to solve?
      2. What class structure would be appropriate for this problem?
      3. What functions and methods will be needed?
      4. How should we test this solution?
      5. What edge cases should we consider?
      
      Outline your solution approach before writing any code.
      """
      
  # Add reference to additional instructions if they exist
  if additional_instructions:
    thinking_prompt += "\n\nMake sure to consider the additional instructions provided, especially any that are relevant to this specific problem."
      
  # Add a thinking prompt
  messages.append(
    HumanMessage(
      content=thinking_prompt
    )
  )
  
  # Get AI's response
  thinking_response = llm.invoke(messages)
  messages.append(thinking_response)
  
  # Log the thinking process
  print("💭 Agent's thinking process:")
  print("-" * 50)
  print(thinking_response.content)
  print("-" * 50)
  
  return {
    **state,
    "messages": messages,
    "next": "write_code"
  }

def write_code(state: State, llm) -> State:
  """Write the solution code in a Python file."""
  print("💻 Agent is writing the solution code...")
  messages = state["messages"]
  
  problem_desc = state.get("problem_description", "")
  
  # Check if we have additional instructions
  additional_instructions = load_instructions()
  
  # Prepare coding prompt
  coding_prompt = f"""
      Create a Python file that implements a solution for: {problem_desc}
      
      Follow these guidelines:
      1. Use a clear class structure with properly named methods
      2. Follow Python best practices and PEP 8 style guidelines
      3. Include the problem description in the class docstring
      4. The class docstring MUST start with: 'Problem: {problem_desc}'
      3. Make sure method names accurately reflect what they do
        - If a method is called filter_even(), it should filter for EVEN numbers
        - If a method is called sort_ascending(), it should sort in ascending order
      4. Include proper type hints
      5. Add descriptive docstrings for the class and all methods
      6. Start the class docstring with a clear statement of the problem being solved: "{problem_desc}"
      7. Handle edge cases appropriately (empty inputs, negative values, etc.)
      8. The file should be saved as solution.py in the current directory
      """
      
  # Add reference to additional instructions if they exist
  if additional_instructions:
    # Extract relevant sections that might apply to code writing
    coding_prompt += """
      
      IMPORTANT: Also follow these additional instructions where applicable:
      - When processing datasets, prefer using pandas for data manipulation
      - Write descriptive variable and function names
      - Handle edge cases explicitly
      - Follow the additional guidelines provided in the instructions
      """
  
  coding_prompt += """
      
      First outline the class structure, then implement the complete solution.
      """
  
  # Add a prompt to write the code
  messages.append(HumanMessage(content=coding_prompt))
  
  # Get AI's response
  code_response = llm.invoke(messages)
  messages.append(code_response)
  
  # Log the code generation process
  print("🔧 Agent's code generation:")
  print("-" * 50)
  print(code_response.content)
  print("-" * 50)
  
  # Extract code from the AI response
  code_blocks = re.findall(r"```python\n(.*?)```", code_response.content, re.DOTALL)
  if code_blocks:
    code = code_blocks[0]
    # Get current directory and create absolute path
    current_dir = get_current_directory.invoke({})
    file_path = os.path.normpath(os.path.join(current_dir, "solution.py"))
    print(f"📂 Current directory: {current_dir}")
    print(f"📄 Writing solution to: {file_path}")
    
    # Use the invoke method with proper parameter format
    write_file.invoke({"file_path": file_path, "content": code})
    print(f"✅ Solution code written to {file_path}")
    
    return {
      **state,
      "messages": messages,
      "code_file": file_path,
      "next": "write_tests"
    }
  else:
    # If no code blocks found, ask for clarification
    print("⚠️ No code block found, asking for clarification...")
    messages.append(
      HumanMessage(
        content="I couldn't find a Python code block in your response. Please provide the solution code in a Python code block (```python)."
      )
    )
    
    clarification_response = llm.invoke(messages)
    messages.append(clarification_response)
    
    # Try to extract code again
    code_blocks = re.findall(r"```python\n(.*?)```", clarification_response.content, re.DOTALL)
    if code_blocks:
      code = code_blocks[0]
      # Get current directory and create absolute path
      current_dir = get_current_directory.invoke({})
      file_path = os.path.normpath(os.path.join(current_dir, "solution.py"))
      print(f"📂 Current directory: {current_dir}")
      print(f"📄 Writing solution to: {file_path}")
      
      write_file.invoke({"file_path": file_path, "content": code})
      print(f"✅ Solution code written to {file_path}")
      
      return {
        **state,
        "messages": messages,
        "code_file": file_path,
        "next": "write_tests"
      }
    else:
      print("❌ Still no code block found, retrying...")
      return {
        **state,
        "messages": messages,
        "next": "write_code"  # Try again
      }

def write_tests(state: State, llm) -> State:
  """Write tests for the solution."""
  print("🧪 Agent is writing tests for the solution...")
  messages = state["messages"]
  
  # Add the code file to the context
  code_content = read_file.invoke({"file_path": state["code_file"]})
  
  # Check if we have additional instructions
  additional_instructions = load_instructions()
  
  # Base testing prompt
  testing_prompt = f"""
      Now that we have the solution code, write comprehensive tests for it.
      The tests should use pytest and be saved as test_solution.py.
      
      Follow these guidelines for creating tests:
      1. Create a clear test class structure with setup_method if needed
      2. Write tests for each method in the solution
      3. Include tests for edge cases: empty lists, negative numbers, large numbers, etc.
      4. Create tests for specific scenarios the solution should handle
      5. Make sure test assertions are clear and specific
      6. Ensure that your test names clearly describe what they're testing
      7. For each test method, include a docstring explaining what it tests
      8. Make sure your tests have clear expected outputs for all assertions
      
      Here's the current solution code for reference:
      ```python
      {code_content}
      ```
      """
      
  # Add reference to additional testing instructions if available
  if additional_instructions and "Testing Guidelines" in additional_instructions:
      testing_prompt += """
      
      Additionally, follow these testing best practices:
      - Write tests that cover both common cases and edge cases
      - Test each method independently with clear assertions
      - Use appropriate test fixtures to set up test environments
      - Verify error handling for invalid inputs
      """
  
  testing_prompt += """
      
      You MUST ensure your test expectations align with the intended functionality of each method.
      For example:
      - If a method is named filter_even(), it should return ONLY even numbers, and tests should verify this
      - If a method is named find_primes(), it should return ONLY prime numbers, and tests should verify this
      - If a method needs to handle negative numbers, write a specific test for that case
      - Double-check that your test assertions match the actual expected behavior of the methods
      """
      
  messages.append(HumanMessage(content=testing_prompt))
  
  # Get AI's response
  test_response = llm.invoke(messages)
  messages.append(test_response)
  
  # Log the test generation process
  print("🔍 Agent's test generation:")
  print("-" * 50)
  print(test_response.content)
  print("-" * 50)
  
  # Extract test code from the AI response
  test_blocks = re.findall(r"```python\n(.*?)```", test_response.content, re.DOTALL)
  if test_blocks:
    test_code = test_blocks[0]
    # Get current directory and create absolute path
    current_dir = get_current_directory.invoke({})
    file_path = os.path.normpath(os.path.join(current_dir, "test_solution.py"))
    print(f"📂 Current directory: {current_dir}")
    print(f"📄 Writing tests to: {file_path}")
    
    write_file.invoke({"file_path": file_path, "content": test_code})
    print(f"✅ Test code written to {file_path}")
    
    return {
      **state,
      "messages": messages,
      "test_file": file_path,
      "next": "run_tests"
    }
  else:
    # If no test blocks found, ask for clarification
    messages.append(
      HumanMessage(
        content="I couldn't find a Python test code block in your response. Please provide the tests in a Python code block (```python)."
      )
    )
    
    clarification_response = llm.invoke(messages)
    messages.append(clarification_response)
    
    # Try to extract test code again
    test_blocks = re.findall(r"```python\n(.*?)```", clarification_response.content, re.DOTALL)
    if test_blocks:
      test_code = test_blocks[0]
      # Get current directory and create absolute path
      current_dir = get_current_directory.invoke({})
      file_path = os.path.normpath(os.path.join(current_dir, "test_solution.py"))
      print(f"📂 Current directory: {current_dir}")
      print(f"📄 Writing tests to: {file_path}")
      
      write_file.invoke({"file_path": file_path, "content": test_code})
      
      return {
        **state,
        "messages": messages,
        "test_file": file_path,
        "next": "run_tests"
      }
    else:
      return {
        **state,
        "messages": messages,
        "next": "write_tests"  # Try again
      }

def execute_tests(state: State) -> State:
  """Run the tests and capture results."""
  print("🏃 Running tests...")
  test_file = state["test_file"]
  
  if not test_file:
    print("❌ No test file specified!")
    return {
      **state,
      "test_results": "ERROR: No test file specified",
      "next": "evaluate_results"
    }
  
  # Run the tests
  test_results = run_tests.invoke({"test_file": test_file})
  
  # Log test results
  print("📊 Test results:")
  print("-" * 50)
  print(test_results)
  print("-" * 50)
  
  return {
    **state,
    "test_results": test_results,
    "next": "evaluate_results"
  }

def evaluate_results(state: State, llm) -> State:
  """Evaluate the test results and decide next steps."""
  print("📈 Agent is evaluating test results...")
  messages = state["messages"]
  test_results = state["test_results"]
  
  # Get the current iteration count, defaulting to 0 if not found, and increment
  try:
    iteration_count = state["iteration_count"] + 1
  except (KeyError, TypeError):
    # Handle case where iteration_count is missing or not an integer
    iteration_count = 1
    print("⚠️ Iteration count not found in state, resetting to 1")
  
  # Update state with new iteration count immediately
  state = {
    **state,
    "iteration_count": iteration_count
  }
  
  print(f"📊 Iteration {iteration_count} of {MAX_ITERATIONS}")
  
  # Extract detailed information about test failures
  failures = extract_test_failures(test_results)
  
  # Format failures for better readability
  formatted_failures = ""
  for test_name, details in failures.items():
    formatted_failures += f"Test: {test_name}\nFailure details:\n{details}\n\n"
  
  # Check if all tests pass
  tests_passed = test_results and "FAILED" not in test_results and "ERROR" not in test_results
  
  # Check if we've reached the maximum iterations
  if iteration_count >= MAX_ITERATIONS or tests_passed:
    # Final state handling - either tests passed or we've reached the maximum iterations
    if tests_passed:
      print("✅ All tests passed! Solution is complete.")
      result_message = "All tests have passed! The solution is complete."
    else:
      print(f"⏰ Maximum iterations ({MAX_ITERATIONS}) reached!")
      result_message = f"We've reached the maximum number of iterations ({MAX_ITERATIONS})."
    
    # Send final summary request
    messages.append(
      HumanMessage(
        content=f"""
        {result_message}
        Here are the final test results:
        
        {test_results}
        
        Please provide a final summary of the solution and any remaining issues.
        """
      )
    )
    
    final_response = llm.invoke(messages)
    messages.append(final_response)
    
    print("📝 Final evaluation:")
    print("-" * 50)
    print(final_response.content)
    print("-" * 50)
    
    return {
      **state,
      "messages": messages,
      "next": "end"
    }
  
  # Add test results to the context with detailed failure information
  messages.append(
    HumanMessage(
      content=f"""
      Here are the test results:
      
      {test_results}
      
      DETAILED TEST FAILURES:
      {formatted_failures if formatted_failures else "No detailed failure information available."}
      
      Analyze these failures carefully by comparing the expected vs. actual output.
      Pay particular attention to:
      
      1. Method names vs. their actual behavior (e.g., a method called filter_even might need to filter for even numbers, not odd)
      2. Edge cases in the tests (negative numbers, special values)
      3. Specific assertions that are failing and exactly how the actual output differs from expected
      4. Any inconsistencies between the test requirements and your implementation
      
      Based on your analysis, determine if:
      1. The solution is correct (all tests pass)
      2. The solution needs improvements (identify specific fixes needed)
      3. There are misunderstandings between method names and their expected behavior
      
      Provide specific implementation fixes for each failing test case.
      """
    )
  )
  
  # Get AI's response
  evaluation_response = llm.invoke(messages)
  messages.append(evaluation_response)
  
  # Log the evaluation
  print("🔍 Agent's evaluation:")
  print("-" * 50)
  print(evaluation_response.content)
  print("-" * 50)
  
  # Check if all tests pass
  if test_results and "FAILED" not in test_results and "ERROR" not in test_results:
    print("✅ All tests passed! Solution is complete.")
    return {
      **state,
      "messages": messages,
      "next": "end"
    }
  else:
    print("🔄 Tests failed, need to improve code...")
    return {
      **state,
      "messages": messages,
      "next": "improve_code"
    }


def improve_code(state: State, llm) -> State:
  """Improve the code based on test results."""
  print("🔧 Agent is improving the code based on test failures...")
  messages = state["messages"]
  code_file = state["code_file"]
  test_file = state["test_file"]
  test_results = state["test_results"]
  iteration_count = state.get("iteration_count", 0)
  
  # Check if we've hit maximum iterations to prevent infinite loops
  max_iterations = CONFIG.get("max_iterations", 8)
  if iteration_count >= max_iterations:
    print(f"⚠️ Maximum iterations reached ({max_iterations}). Breaking the improvement loop.")
    return {
      **state,
      "messages": messages,
      "next": "end"
    }
  
  # Track iteration counts to prevent infinite loops
  print(f"📊 Current iteration: {iteration_count + 1} of {max_iterations}")
  
  if not code_file:
    print("❌ No code file specified!")
    return {
      **state,
      "messages": messages,
      "next": "write_code"
    }
  
  if not test_file:
    print("❌ No test file specified!")
    return {
      **state,
      "messages": messages,
      "next": "write_tests"
    }
  
  # Get the current code and test content
  code_content = read_file.invoke({"file_path": code_file})
  test_content = read_file.invoke({"file_path": test_file})
  
  # Extract detailed test failure information
  failures = extract_test_failures(test_results)
  
  # Format failures for better readability
  formatted_failures = ""
  for test_name, details in failures.items():
    formatted_failures += f"Test: {test_name}\nFailure details:\n{details}\n\n"
  
  # Extract failing test names for more focused fixes
  failing_tests = list(failures.keys())
  failing_tests_str = "\n".join(failing_tests) if failing_tests else "Unknown test failures"
  
  # Check for additional instructions
  additional_instructions = load_instructions()
  
  # Base improvement prompt
  improvement_prompt = f"""
      Let's fix the failing tests in the solution. Here are the specific failing tests:
      
      {failing_tests_str}
      
      DETAILED TEST FAILURES:
      {formatted_failures if formatted_failures else "No detailed failure information available."}
      
      Current solution code:
      ```python
      {code_content}
      ```
      
      Current test code:
      ```python
      {test_content}
      ```
      
      IMPORTANT GUIDELINES FOR FIXING THE CODE:
      1. Carefully analyze each test failure to understand the issue
      2. Make sure method names match their behavior:
         - If a method is named filter_even(), it MUST return EVEN numbers only
         - If a method is named find_primes(), it MUST return ONLY prime numbers
      3. Pay attention to test expectations - your code must satisfy the tests
      4. Check specific issues shown in the test failures (examine expected vs. actual values)
      5. Look for edge cases the tests are checking (negative numbers, empty inputs, etc.)
      6. Ensure your solution handles all edge cases included in the tests
      7. Keep the class docstring that includes the problem description
      8. When in doubt, prioritize making the tests pass over keeping the current implementation
      """
      
  # Add reference to additional instructions if available
  if additional_instructions:
      improvement_prompt += """
      
      Also remember to follow best practices from the additional instructions, including:
      - Use clear and descriptive variable names
      - Include proper type hints
      - Handle edge cases explicitly
      - Follow PEP 8 style guidelines
      - If working with data, use pandas efficiently
      """
      
  improvement_prompt += """
      
      Please provide a COMPLETE improved version of the solution code that will pass all tests.
      """
      
  messages.append(HumanMessage(content=improvement_prompt))
  
  # Get AI's response
  improved_response = llm.invoke(messages)
  messages.append(improved_response)
  
  # Extract improved code from the AI response
  improved_code_blocks = re.findall(r"```python\n(.*?)```", improved_response.content, re.DOTALL)
  if improved_code_blocks:
    improved_code = improved_code_blocks[0]
    # Check if code_file is a full path
    if not os.path.isabs(code_file):
      current_dir = get_current_directory.invoke({})
      code_file = os.path.normpath(os.path.join(current_dir, code_file))
      print(f"📂 Using absolute path for code file: {code_file}")
    
    # Write the improved solution file
    write_file.invoke({"file_path": code_file, "content": improved_code})
    
    return {
      **state,
      "messages": messages,
      "next": "run_tests"
    }
  else:
    # If no improved code blocks found, ask for clarification
    messages.append(
      HumanMessage(
        content="I couldn't find an improved Python code block in your response. Please provide the improved solution code in a Python code block (```python)."
      )
    )
    
    clarification_response = llm.invoke(messages)
    messages.append(clarification_response)
    
    # Try to extract improved code again
    improved_code_blocks = re.findall(r"```python\n(.*?)```", clarification_response.content, re.DOTALL)
    if improved_code_blocks:
      improved_code = improved_code_blocks[0]
      # Check if code_file is a full path
      if not os.path.isabs(code_file):
        current_dir = get_current_directory.invoke({})
        code_file = os.path.normpath(os.path.join(current_dir, code_file))
        print(f"📂 Using absolute path for code file: {code_file}")
      
      # Write the improved solution file
      write_file.invoke({"file_path": code_file, "content": improved_code})
      
      return {
        **state,
        "messages": messages,
        "next": "run_tests"
      }
    else:
      return {
        **state,
        "messages": messages,
        "next": "improve_code"  # Try again
      }

def router(state: State) -> str:
  """
  Route to the next node based on the state.
  
  This function also enforces the MAX_ITERATIONS limit, ensuring the agent stops
  after reaching the configured maximum number of iterations, even if the tests
  are still failing.
  """
  # Check iteration count to enforce max iterations
  iteration_count = state.get("iteration_count", 0)
  if iteration_count >= MAX_ITERATIONS:
    print(f"⚠️ Maximum iterations reached ({iteration_count} of {MAX_ITERATIONS}). Forcing end state.")
    return "end"
    
  # Get the next step from the state
  next_step = state.get("next")
  if next_step is None:
    return "end"  # Default fallback
    
  return next_step


# Build the agent graph
def build_agent(checkpointer=None):
  """Build and return the agent graph."""
  # Get graph settings from config
  graph_settings = CONFIG.get("graph_settings", {})
  recursion_limit = graph_settings.get("recursion_limit", 50)
  
  # Create StateGraph (note: recursion_limit handling is done separately in newer versions)
  builder = StateGraph(State)
  
  # Add nodes
  builder.add_node("initialize", initialize_state)
  builder.add_node("think", lambda state: think(state, create_llm()))
  builder.add_node("write_code", lambda state: write_code(state, create_llm()))
  builder.add_node("write_tests", lambda state: write_tests(state, create_llm()))
  builder.add_node("run_tests", execute_tests)
  builder.add_node("evaluate_results", lambda state: evaluate_results(state, create_llm()))
  builder.add_node("improve_code", lambda state: improve_code(state, create_llm()))
  
  # Add edges
  builder.add_edge("initialize", "think")
  builder.add_edge("think", "write_code")
  builder.add_edge("write_code", "write_tests")
  builder.add_edge("write_tests", "run_tests")
  builder.add_edge("run_tests", "evaluate_results")
  builder.add_edge("improve_code", "run_tests")
  
  # Add conditional edges from router
  builder.add_conditional_edges(
    "evaluate_results",
    router,
    {
      "end": END,
      "improve_code": "improve_code"
    }
  )
  
  # Set entry point
  builder.set_entry_point("initialize")
  
  # Get graph settings from config
  graph_settings = CONFIG.get("graph_settings", {})
  recursion_limit = graph_settings.get("recursion_limit", 50)
  max_retries = graph_settings.get("max_retries", 3)
  
  print(f"🔧 Building agent graph (recursion_limit={recursion_limit} configured but handled differently in newer LangGraph)")
  
  # In newer versions of LangGraph, recursion_limit may be handled differently
  # We'll compile without additional parameters to ensure compatibility
  
  # Create the graph with optional checkpointer
  if checkpointer:
    return builder.compile(checkpointer=checkpointer)
  else:
    return builder.compile()


# Main function to run the agent
def main():
  """Main function to run the coding agent."""
  parser = argparse.ArgumentParser(description="Coding Agent built with LangGraph")
  parser.add_argument(
    "problem",
    type=str,
    help="Description of the problem to solve"
  )
  parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Expected output or examples (optional)"
  )
  # Add parameters from config file that can be overridden
  llm_settings = CONFIG.get("llm_settings", {})
  parser.add_argument(
    "--provider",
    type=str,
    default=llm_settings.get("default_provider", "openai"),
    choices=["openai", "bedrock"],
    help="LLM provider to use (openai or bedrock)"
  )
  parser.add_argument(
    "--model",
    type=str,
    default=llm_settings.get("default_model", "gpt-4o-mini"),
    help="Model name to use"
  )
  parser.add_argument(
    "--temperature",
    type=float,
    default=llm_settings.get("default_temperature", 0.0),
    help="Temperature for generation"
  )
  parser.add_argument(
    "--max-iterations",
    type=int,
    default=CONFIG.get("max_iterations", 8),
    help="Maximum number of iterations for code improvement"
  )
  parser.add_argument(
    "--config",
    type=str,
    default=CONFIG_FILE,
    help="Path to configuration file"
  )
  
  args = parser.parse_args()
  
  # Update global MAX_ITERATIONS with command line value
  global MAX_ITERATIONS
  MAX_ITERATIONS = args.max_iterations
  
  print("🤖 Initializing Coding Agent...")
  print(f"🔧 Using model: {args.model}")
  print(f"🔧 Using provider: {args.provider}")
  print(f"🔧 Temperature: {args.temperature}")
  print(f"🔧 Max iterations: {MAX_ITERATIONS}")
  
  # Check for OpenAI API key if using OpenAI
  if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable not set")
    print("Please set your OpenAI API key by running:")
    print("export OPENAI_API_KEY=your_api_key (on Linux/Mac)")
    print("set OPENAI_API_KEY=your_api_key (on Windows)")
    return
  
  # Update the LLM creator with user preferences
  global create_llm
  original_create_llm = create_llm
  
  # Define a new create_llm function with the user's parameters
  def new_create_llm(temperature=None, model_name=None, provider=None):
    return original_create_llm(
      temperature=args.temperature,
      model_name=args.model,
      provider=args.provider
    )
  
  create_llm = new_create_llm
  
  print("🔄 Building agent graph...")
  
  # Clean up any existing solution files
  clean_old_solution_files()
  
  # Check if checkpointing is enabled in config
  use_checkpointing = CONFIG.get("graph_settings", {}).get("use_checkpointing", True)
  
  try:
    if use_checkpointing:
      try:
        # Initialize the MemorySaver for in-memory checkpointing
        print("📁 Using in-memory checkpointing")
        
        # Initialize the MemorySaver checkpointer with configurable keys
        # In newer versions of LangGraph, the MemorySaver expects configurable keys
        # We'll create a MemorySaver without checkpointing and just run the agent directly
        print("ℹ️ In-memory checkpointing requires additional configuration in newer LangGraph versions")
        print("ℹ️ Running agent without checkpointing")
        
        # Build the agent without a checkpointer
        agent = build_agent()
        print("✅ Agent graph built successfully without checkpointing")
      except Exception as e:
        print(f"⚠️ Error initializing checkpointer: {e}")
        print("⚠️ Falling back to no checkpointing")
        agent = build_agent()
    else:
      print("ℹ️ Checkpointing disabled in configuration")
      agent = build_agent()
  except Exception as e:
    print(f"❌ Error building agent: {e}")
    print("⚠️ Falling back to building agent without checkpointer...")
    try:
      agent = build_agent(None)
      print("✅ Agent graph built successfully without checkpointing")
    except Exception as e:
      print(f"❌ Error building agent: {e}")
      return
  
  # Create initial state
  initial_state = {
    "messages": [],
    "problem_description": args.problem,
    "expected_output": args.output,
    "code_file": None,
    "test_file": None,
    "test_results": None,
    "iteration_count": 0,
    "next": None
  }
  
  # Run the agent with event tracer for logging
  print(f"🧠 Working on problem: {args.problem}")
  
  # Stream the graph execution
  final_state = None
  
  try:
    print("🚀 Starting agent execution...")
    
    # Get graph settings from config
    graph_settings = CONFIG.get("graph_settings", {})
    recursion_limit = graph_settings.get("recursion_limit", 50)
    
    print(f"📊 Using configuration (recursion_limit: {recursion_limit}, max_iterations: {MAX_ITERATIONS})...")
    all_step_outputs = []
    
    try:
      # Stream execution with memory persistence via checkpointer
      print("🔄 Starting execution stream with memory checkpointing...")
      
      # Create a unique identifier for this run
      run_id = f"run_{int(time.time())}"
      print(f"🧵 Using run ID: {run_id}")
      
      # Note: We don't pass thread_id directly as it's handled by the checkpointer
      for step_output in agent.stream(initial_state):
        # Store each output to debug later if needed
        all_step_outputs.append(step_output)
        
        # Each step_output is a dictionary with node names as keys
        for node_name, node_state in step_output.items():
          print(f"🔄 Completed step: {node_name}")
          final_state = node_state
          
          # Check if we've reached the maximum iterations to provide clearer feedback
          if node_name == "evaluate_results" and final_state.get("iteration_count", 0) >= MAX_ITERATIONS:
            print(f"⏰ Maximum iterations ({MAX_ITERATIONS}) reached! Agent will finish soon.")
      
      print("🎯 Agent execution completed successfully")
    except ValueError as e:
      if "Checkpointer" in str(e):
        print(f"❌ Error with checkpointer: {e}")
        print("⚠️ Checkpointing not supported in this version of LangGraph without additional configuration.")
        print("❌ Please update agent.py to remove checkpointing or configure LangGraph properly.")
        return
      else:
        raise
    except Exception as e:
      # More gracefully handle the recursion limit error
      if "recursion_limit" in str(e).lower():
        print(f"⚠️ Warning: {str(e)}")
        print("💡 The agent reached the recursion limit. This often happens when handling complex problems.")
        print("💡 Try increasing the recursion_limit in config.json or reducing the complexity of the problem.")
        # Save the last state if available
        if all_step_outputs:
          last_output = all_step_outputs[-1]
          for node_name, node_state in last_output.items():
            final_state = node_state
            print(f"📊 Using last successful state from step: {node_name}, iteration: {node_state.get('iteration_count', 0)}")
      else:
        raise
  except Exception as e:
    print(f"❌ Error during agent execution: {e}")
    import traceback
    traceback.print_exc()
    
    # Handle specific LangGraph errors
    if "GraphRecursionError" in str(e) and "recursion_limit" in str(e):
      print("\n⚠️ The agent reached the recursion limit. This usually happens when:")
      print("  1. The problem is complex and needs more iterations")
      print("  2. The agent is stuck in a loop trying the same approach")
      print("\n💡 Try increasing the recursion_limit in config.json")
    
    return
  
  print("\n" + "=" * 50)
  print("🎉 Solution Process Complete!")
  print("=" * 50)
  
  if final_state:
    print(f"📄 Solution file: {final_state.get('code_file', 'N/A')}")
    print(f"🧪 Test file: {final_state.get('test_file', 'N/A')}")
    
    # Summary of test results
    if final_state.get("test_results"):
      print("\n🧪 Final Test Results:")
      print("-" * 50)
      print(final_state["test_results"])
    
    print("\n📝 Summary:")
    test_results = final_state.get("test_results", "")
    if final_state.get("next") == "end" and "ERROR" not in test_results and "FAILED" not in test_results:
      print("✅ All tests passed. The solution is complete.")
    else:
      print(f"⚠️ Solution completed with {final_state.get('iteration_count', 0)} iterations but some tests may be failing.")
  else:
    print("❌ No final state available.")
  
  print("\nYou can find the solution in 'solution.py' and the tests in 'test_solution.py'.")


def clean_old_solution_files():
  """
  Clean up any existing solution and test files before running the agent.
  This helps prevent conflicts with previous runs.
  """
  print("🧹 Cleaning up any existing solution files...")
  try:
    current_dir = get_current_directory.invoke({})
    solution_path = os.path.normpath(os.path.join(current_dir, "solution.py"))
    test_path = os.path.normpath(os.path.join(current_dir, "test_solution.py"))
    
    if os.path.exists(solution_path):
      os.remove(solution_path)
      print(f"✅ Removed existing solution file: {solution_path}")
    
    if os.path.exists(test_path):
      os.remove(test_path)
      print(f"✅ Removed existing test file: {test_path}")
  except Exception as e:
    print(f"⚠️ Warning during cleanup: {str(e)}")


def extract_test_failures(test_output: Optional[str]) -> dict:
  """
  Extract detailed information about failed tests from pytest output.
  
  Args:
    test_output: The output from pytest
    
  Returns:
    A dictionary mapping test names to failure details
  """
  if not test_output:
    return {}
  
  failures = {}
  # Match patterns like "test_something ..." or "TestClass.test_something ..."
  test_blocks = re.findall(r'_{10,}\s+(.+?)\s+_{10,}(.*?)(?=_{10,}|\Z)', test_output, re.DOTALL)
  
  for test_name, details in test_blocks:
    # Only include if it contains a failure
    if "FAILED" in details or "ERROR" in details:
      # Extract clean test name
      clean_name = test_name.split('::')[-1].strip()
      # Store details with some cleanup
      failures[clean_name] = details.strip()
  
  return failures


if __name__ == "__main__":
  clean_old_solution_files()
  main()
