#!/usr/bin/env python3
"""
Test cases for the LangGraph Coding Agent
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import importlib.util

# Import the agent module
agent_path = os.path.join(os.path.dirname(__file__), 'agent.py')
spec = importlib.util.spec_from_file_location("agent", agent_path)
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)


class TestCodingAgent(unittest.TestCase):
    """Test cases for the coding agent."""

    def test_tool_read_file(self):
        """Test the read_file tool."""
        # Create a temporary file
        with open("temp_test.txt", "w") as f:
            f.write("Test content")
        
        # Test reading the file
        result = agent_module.read_file("temp_test.txt")
        self.assertEqual(result, "Test content")
        
        # Clean up
        os.remove("temp_test.txt")
    
    def test_tool_write_file(self):
        """Test the write_file tool."""
        # Write to a file
        agent_module.write_file("temp_test_write.txt", "Written content")
        
        # Verify the content
        with open("temp_test_write.txt", "r") as f:
            content = f.read()
        
        self.assertEqual(content, "Written content")
        
        # Clean up
        os.remove("temp_test_write.txt")
    
    def test_state_initialization(self):
        """Test the state initialization."""
        initial_state = {
            "messages": [],
            "problem_description": "Test problem",
            "expected_output": "Test output",
            "code_file": None,
            "test_file": None,
            "test_results": None,
            "iteration_count": 0,
            "next": None
        }
        
        result_state = agent_module.initialize_state(initial_state)
        
        # Check that messages were added
        self.assertTrue(len(result_state["messages"]) >= 2)
        # Check that next was set to "think"
        self.assertEqual(result_state["next"], "think")
    
    @patch("agent_module.ChatOpenAI")
    def test_build_agent(self, mock_chat):
        """Test building the agent."""
        # Mock the LLM
        mock_chat.return_value = MagicMock()
        
        # Build the agent
        agent = agent_module.build_agent()
        
        # Verify that the agent is a compiled StateGraph
        self.assertIsNotNone(agent)


if __name__ == "__main__":
    unittest.main()
