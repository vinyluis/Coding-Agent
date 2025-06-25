"""
Example usage of the coding agent to solve a simple problem:
Calculate the factorial of a number.
"""

import os
import sys
import subprocess

def main():
    """Run the coding agent with a sample problem."""
    problem = "Create a class that calculates the factorial of a positive integer"
    
    try:
        # Path to the agent.py file
        agent_path = os.path.join(os.path.dirname(__file__), 'agent.py')
        
        print("Installing required dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                              os.path.join(os.path.dirname(__file__), "requirements.txt")])
        
        print("\nRunning coding agent...")
        cmd = [sys.executable, agent_path, problem]
        subprocess.run(cmd)
        
    except Exception as e:
        print(f"Error running the agent: {str(e)}")

if __name__ == "__main__":
    main()
