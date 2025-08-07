#!/usr/bin/env python3
"""
Test script for the modular reward functions
"""

import re

def extract_xml_answer(completion):
    """Extract raw text between <answer> tags"""
    match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def check_strict_formatting(completion):
    """Check for proper formatting with newlines"""
    # Check for proper multi-line format with newlines
    reasoning_pattern = r'<reasoning>\s*\n.*?\n\s*</reasoning>'
    answer_pattern = r'<answer>\s*\n.*?\n\s*</answer>'
    
    has_proper_reasoning = bool(re.search(reasoning_pattern, completion, re.DOTALL))
    has_proper_answer = bool(re.search(answer_pattern, completion, re.DOTALL))
    
    if has_proper_reasoning and has_proper_answer:
        return 1.0
    return 0.0

def check_soft_formatting(completion):
    """Check for basic XML tags presence"""
    has_reasoning = "<reasoning>" in completion and "</reasoning>" in completion
    has_answer = "<answer>" in completion and "</answer>" in completion
    
    if has_reasoning and has_answer:
        return 1.0  # Same score as strict formatting
    elif has_reasoning or has_answer:
        return 0.3  # Partial credit
    return -0.5  # No proper formatting

def check_integer_reward(completion):
    """Reward integer answers - simplified version"""
    answer = extract_xml_answer(completion)
    if answer is None:
        return 0.0  # No extractable answer
        
    # Clean the answer (remove extra spaces, etc.)
    answer_clean = answer.strip()
    
    # Check if it's a valid integer (very simple)
    if answer_clean.lstrip('-').isdigit():
        return 0.5  # Good integer answer
    
    return 0.0  # Not an integer

def check_correctness(prompt, completion):
    """Compare extracted answer against correct answer encoded in prompt"""
    # Extract correct answer from prompt (encoded as [CORRECT:123])
    correct_match = re.search(r'\[CORRECT:([^\]]+)\]', prompt)
    if not correct_match:
        return 0.0  # No correct answer provided
        
    correct_answer = correct_match.group(1).strip()
    extracted_answer = extract_xml_answer(completion)
    
    if extracted_answer is None:
        return -0.5  # No answer extracted
        
    try:
        # Clean both answers for comparison (remove $ and commas)
        correct_clean = correct_answer.replace("$", "").replace(",", "").strip()
        extracted_clean = extracted_answer.replace("$", "").replace(",", "").strip()
        
        # Try to compare as numbers
        if correct_clean.replace(".", "").replace("-", "").isdigit() and extracted_clean.replace(".", "").replace("-", "").isdigit():
            if float(correct_clean) == float(extracted_clean):
                return 1.0  # Correct answer!
            else:
                return -0.5  # Wrong answer
        else:
            # Compare as strings if not numbers
            if correct_clean.lower() == extracted_clean.lower():
                return 1.0  # Correct answer!
            else:
                return -0.5  # Wrong answer
    except ValueError:
        return -0.3  # Couldn't parse for comparison

# Test cases
test_cases = [
    {
        "name": "Perfect response",
        "prompt": "What is 5 + 3? [CORRECT:8]",
        "completion": "<reasoning>\nI need to add 5 + 3 = 8\n</reasoning>\n<answer>8</answer>",
        "expected_scores": {"soft_format": 1.0, "integer": 0.5, "correctness": 1.0}
    },
    {
        "name": "Wrong answer",
        "prompt": "What is 5 + 3? [CORRECT:8]", 
        "completion": "<reasoning>\nI need to add 5 + 3 = 9\n</reasoning>\n<answer>9</answer>",
        "expected_scores": {"soft_format": 1.0, "integer": 0.5, "correctness": -0.5}
    },
    {
        "name": "GSM8K format with dollar",
        "prompt": "How much money? [CORRECT:$10]",
        "completion": "<reasoning>\nThe calculation shows $10\n</reasoning>\n<answer>$10</answer>",
        "expected_scores": {"soft_format": 1.0, "integer": 0.0, "correctness": 1.0}
    },
    {
        "name": "No formatting",
        "prompt": "What is 5 + 3? [CORRECT:8]",
        "completion": "The answer is 8",
        "expected_scores": {"soft_format": -0.5, "integer": -0.2, "correctness": -0.5}
    },
    {
        "name": "Float answer",
        "prompt": "What is 5 + 3? [CORRECT:8]",
        "completion": "<reasoning>\nCalculation\n</reasoning>\n<answer>8.0</answer>",
        "expected_scores": {"soft_format": 1.0, "integer": -0.3, "correctness": -0.3}
    }
]

if __name__ == "__main__":
    print("Testing modular reward functions...\n")
    
    for test_case in test_cases:
        print(f"Test: {test_case['name']}")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Completion: {test_case['completion']}")
        
        # Calculate actual scores
        soft_score = check_soft_formatting(test_case['completion'])
        integer_score = check_integer_reward(test_case['completion'])
        correctness_score = check_correctness(test_case['prompt'], test_case['completion'])
        total_score = soft_score + integer_score + correctness_score
        
        print(f"Scores:")
        print(f"  Soft formatting: {soft_score}")
        print(f"  Integer reward: {integer_score}")
        print(f"  Correctness: {correctness_score}")
        print(f"  Total: {total_score}")
        print(f"  Extracted answer: '{extract_xml_answer(test_case['completion'])}'")
        print()
