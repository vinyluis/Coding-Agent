#!/usr/bin/env python3
"""
LangGraph Coding Agent that solves programming problems by:
1. Generating code in a class structure
2. Creating tests to validate the solution
3. Iterating on the code based on test results
"""

import os
import re
import subprocess
import argparse
import os.path as ospath
from typing import TypedDict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
try:
  from langchain_aws import ChatBedrock
except ImportError:
  # Optional dependency, will be None if not installed
  ChatBedrock = None
from langchain_core.tools import ToolException, tool
from langgraph.graph import StateGraph, END

# Configure maximum iterations to prevent infinite loops
MAX_ITERATIONS = 3


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
  return """You are a professional coding agent that solves programming problems.
  
You are tasked with solving a problem by:
1. Understanding the requirements
2. Creating a solution in a Python class structure
3. Creating tests for the solution
4. Improving the code based on test results

You have access to tools that allow you to read/write files, run commands, search text, run tests, and more.
Be thorough in your analysis and methodical in your approach.
"""

def initialize_state(state: State) -> State:
  """Initialize the agent state with the problem description."""
  messages = state["messages"]
  
  # Extract problem description and expected output
  problem_description = state.get("problem_description", "")
  expected_output = state.get("expected_output", None)
  
  # Add system message with the agent's instructions
  messages.append(SystemMessage(content=get_system_prompt()))
  
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
  
  # Add a thinking prompt
  messages.append(
    HumanMessage(
      content="""
      Think through this problem step by step:
      1. What is the core problem to solve?
      2. What class structure would be appropriate for this problem?
      3. What functions and methods will be needed?
      4. How should we test this solution?
      5. What edge cases should we consider?
      
      Outline your solution approach before writing any code.
      """
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
  
  # Add a prompt to write the code
  messages.append(
    HumanMessage(
      content="""
      Create a Python file that implements your solution.
      Use a class structure and follow best practices for Python code.
      The file should be saved as solution.py in the current directory.
      
      Please outline the code structure before writing the full implementation.
      """
    )
  )
  
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
  
  messages.append(
    HumanMessage(
      content=f"""
      Now that we have the solution code, write comprehensive tests for it.
      The tests should use pytest and be saved as test_solution.py.
      Make sure to test all aspects of the solution, including edge cases.
      
      Here's the current solution code for reference:
      ```python
      {code_content}
      ```
      """
    )
  )
  
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
  iteration_count = state["iteration_count"] + 1
  
  # Check if we've reached the maximum iterations
  if iteration_count >= MAX_ITERATIONS:
    print(f"⏰ Maximum iterations ({MAX_ITERATIONS}) reached!")
    messages.append(
      HumanMessage(
        content=f"""
        We've reached the maximum number of iterations ({MAX_ITERATIONS}).
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
      "iteration_count": iteration_count,
      "next": "end"
    }
  
  # Add test results to the context
  messages.append(
    HumanMessage(
      content=f"""
      Here are the test results:
      
      {test_results}
      
      Based on these results, determine if:
      1. The solution is correct (all tests pass)
      2. The solution needs improvements
      3. The tests need to be fixed
      
      Please analyze the results and suggest next steps.
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
      "iteration_count": iteration_count,
      "next": "end"
    }
  else:
    print("🔄 Tests failed, need to improve code...")
    return {
      **state,
      "messages": messages,
      "iteration_count": iteration_count,
      "next": "improve_code"
    }

def improve_code(state: State, llm) -> State:
  """Improve the code based on test results."""
  print("🔧 Agent is improving the code based on test failures...")
  messages = state["messages"]
  code_file = state["code_file"]
  test_file = state["test_file"]
  
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
  
  messages.append(
    HumanMessage(
      content=f"""
      Let's improve the code to fix the failing tests.
      
      Current solution code:
      ```python
      {code_content}
      ```
      
      Current test code:
      ```python
      {test_content}
      ```
      
      Please provide an improved version of the solution code.
      """
    )
  )
  
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
  """Route to the next node based on the state."""
  next_step = state.get("next")
  if next_step is None:
    return "end"  # Default fallback
  return next_step

# Build the agent graph
def build_agent(checkpointer=None):
  """Build and return the agent graph."""
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
  parser.add_argument(
    "--provider",
    type=str,
    default="openai",
    choices=["openai", "bedrock"],
    help="LLM provider to use (openai or bedrock)"
  )
  parser.add_argument(
    "--model",
    type=str,
    default="gpt-4o-mini",
    help="Model name to use"
  )
  parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="Temperature for generation"
  )
  
  args = parser.parse_args()
  
  print("🤖 Initializing Coding Agent...")
  print(f"🔧 Using model: {args.model}")
  print(f"🔧 Using provider: {args.provider}")
  print(f"🔧 Temperature: {args.temperature}")
  
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
  
  # Build the agent
  try:
    # Don't use checkpointer at all, it's causing issues
    agent = build_agent(None)
    print("✅ Agent graph built successfully")
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
    
    # Create a simplified config that doesn't use checkpointer
    config = None
    
    print("🔄 Starting stream execution...")
    all_step_outputs = []
    
    try:
      for step_output in agent.stream(initial_state, config=config):
        # Store each output to debug later if needed
        all_step_outputs.append(step_output)
        
        # Each step_output is a dictionary with node names as keys
        for node_name, node_state in step_output.items():
          print(f"🔄 Completed step: {node_name}")
          final_state = node_state
      
      print("🎯 Agent execution completed successfully")
    except ValueError as e:
      if "Checkpointer" in str(e):
        print(f"❌ Error with checkpointer: {e}")
        print("Attempting to run without config...")
        # Try again without any config
        for step_output in agent.stream(initial_state):
          # Each step_output is a dictionary with node names as keys
          for node_name, node_state in step_output.items():
            print(f"🔄 Completed step: {node_name}")
            final_state = node_state
      else:
        raise
  except Exception as e:
    print(f"❌ Error during agent execution: {e}")
    import traceback
    traceback.print_exc()
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


if __name__ == "__main__":
  clean_old_solution_files()
  main()
