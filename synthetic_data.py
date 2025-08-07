"""
Synthetic Math Dataset Generator for GRPO Training
Generates diverse math problems with varying response quality for group-relative optimization
"""

import random
import json


def generate_synthetic_dataset():
    """Generate synthetic math dataset with diverse problems and response quality"""
    
    # Problem templates by difficulty
    easy_problems = [
        "What is {a} + {b}?",
        "What is {a} - {b}?", 
        "If Sarah has {a} apples and eats {b}, how many remain?"
    ]
    
    medium_problems = [
        "What is {a} × {b} + {c}?",
        "If a book costs ${a}.{b:02d} and you pay with ${c}, what's your change?",
        "A rectangle has length {a} and width {b}. What is its area?"
    ]
    
    hard_problems = [
        "If {a} workers can build {b} houses in {c} days, how many days for {d} workers to build {e} houses?",
        "What is ({a} + {b}) × {c} - {d} ÷ {e}?",
        "A train travels {a} km/h for {b} hours, then {c} km/h for {d} hours. What's the total distance?"
    ]
    
    def generate_problem_set(templates, difficulty, count=20):
        problems = []
        for _ in range(count):
            template = random.choice(templates)
            
            if difficulty == "easy":
                values = {chr(97+i): random.randint(1, 20) for i in range(5)}
                # Ensure subtraction doesn't go negative
                if "- {b}" in template:
                    values['b'] = random.randint(1, values['a'])
            elif difficulty == "medium":
                values = {chr(97+i): random.randint(10, 99) for i in range(5)}
                if "change" in template:
                    values['c'] = values['a'] + random.randint(1, 50)
            else:  # hard
                values = {chr(97+i): random.randint(5, 50) for i in range(5)}
                
            problem = template.format(**values)
            correct_answer = calculate_answer(template, values)
            problems.append({
                "problem": problem,
                "answer": correct_answer,
                "difficulty": difficulty,
                "values": values
            })
        return problems
    
    def calculate_answer(template, values):
        """Calculate correct answer for problem templates"""
        if "+" in template and "×" not in template:
            return values['a'] + values['b']
        elif "-" in template:
            return values['a'] - values['b']
        elif "×" in template and "+" in template:
            return values['a'] * values['b'] + values['c']
        elif "area" in template:
            return values['a'] * values['b']
        elif "change" in template:
            return values['c'] - values['a']
        elif "workers" in template:
            # Rate calculation: (workers1 * houses2 * days1) / (houses1 * workers2)
            return int((values['a'] * values['e'] * values['c']) / (values['b'] * values['d']))
        elif "train" in template:
            return values['a'] * values['b'] + values['c'] * values['d']
        elif "÷" in template:
            # Order of operations: (a + b) × c - d ÷ e
            return int((values['a'] + values['b']) * values['c'] - values['d'] / values['e'])
        return 42  # Placeholder
    
    def generate_response_variants(problem_data):
        """Generate 5 response variants with different quality levels"""
        problem = problem_data["problem"]
        correct_answer = problem_data["answer"]
        
        # Perfect response
        perfect = f"""<reasoning>
Let me solve this step by step:
{generate_perfect_reasoning(problem_data)}
</reasoning>
<answer>{correct_answer}</answer>"""
        
        # Good format, wrong math
        wrong_answer = correct_answer + random.randint(-5, 5)
        if wrong_answer == correct_answer:
            wrong_answer += 1
        format_wrong = f"""<reasoning>
{generate_flawed_reasoning(problem_data)}
</reasoning>
<answer>{wrong_answer}</answer>"""
        
        # Correct answer, poor format
        correct_ugly = f"The answer is {correct_answer}. I calculated it quickly."
        
        # Partial reasoning, format issues
        partial = f"""<reasoning>
{generate_partial_reasoning(problem_data)}
</reasoning>
The final answer is {correct_answer}"""
        
        # Completely wrong
        completely_wrong = f"I think the answer is {random.randint(1, 100)}."
        
        return [
            {"response": perfect, "quality_score": 1.0},
            {"response": format_wrong, "quality_score": 0.3},
            {"response": correct_ugly, "quality_score": 0.6}, 
            {"response": partial, "quality_score": 0.7},
            {"response": completely_wrong, "quality_score": 0.1}
        ]
    
    def generate_perfect_reasoning(problem_data):
        """Generate step-by-step reasoning"""
        problem = problem_data["problem"]
        values = problem_data["values"]
        
        if "+" in problem and "×" not in problem and "÷" not in problem:
            a, b = values['a'], values['b']
            return f"I need to add {a} + {b} = {a + b}"
        elif "-" in problem:
            a, b = values['a'], values['b']
            return f"I need to subtract {a} - {b} = {a - b}"
        elif "×" in problem and "+" in problem:
            a, b, c = values['a'], values['b'], values['c']
            return f"First multiply {a} × {b} = {a * b}, then add {c}: {a * b} + {c} = {a * b + c}"
        elif "area" in problem:
            a, b = values['a'], values['b']
            return f"Area = length × width = {a} × {b} = {a * b}"
        elif "change" in problem:
            a, c = values['a'], values['c']
            return f"Change = amount paid - cost = ${c} - ${a} = ${c - a}"
        return "Working through this problem systematically..."
    
    def generate_flawed_reasoning(problem_data):
        """Generate reasoning with logical errors"""
        flawed_options = [
            "I'll solve this by adding all the numbers together, which gives me the result.",
            "I think I need to multiply everything by 2 to get the right answer.",
            "Let me just estimate this - it looks like the answer should be around this number."
        ]
        return random.choice(flawed_options)
    
    def generate_partial_reasoning(problem_data):
        """Generate incomplete reasoning"""
        partial_options = [
            "First I identify the operation needed...",
            "Looking at this problem, I can see that...",
            "Let me start by figuring out what we're solving for..."
        ]
        return random.choice(partial_options)
    
    # Generate dataset
    dataset = []
    
    # Generate problems for each difficulty
    easy_probs = generate_problem_set(easy_problems, "easy", 15)
    medium_probs = generate_problem_set(medium_problems, "medium", 15) 
    hard_probs = generate_problem_set(hard_problems, "hard", 10)
    
    all_problems = easy_probs + medium_probs + hard_probs
    
    # Create GRPO groups
    for problem_data in all_problems:
        variants = generate_response_variants(problem_data)
        
        dataset.append({
            "problem": problem_data["problem"],
            "correct_answer": problem_data["answer"],
            "difficulty": problem_data["difficulty"],
            "responses": variants
        })
    
    return dataset


def validate_dataset_diversity(dataset):
    """Validate that the dataset has good coverage across quality dimensions"""
    
    # Count by difficulty
    difficulty_counts = {}
    format_quality_counts = {"perfect": 0, "partial": 0, "poor": 0}
    
    for entry in dataset:
        diff = entry["difficulty"]
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        # Analyze response quality distribution
        for response in entry["responses"]:
            if "<reasoning>" in response["response"] and "<answer>" in response["response"]:
                format_quality_counts["perfect"] += 1
            elif "<reasoning>" in response["response"] or "<answer>" in response["response"]:
                format_quality_counts["partial"] += 1
            else:
                format_quality_counts["poor"] += 1
    
    print(f"Dataset diversity validation:")
    print(f"Difficulty distribution: {difficulty_counts}")
    print(f"Format quality distribution: {format_quality_counts}")
    print(f"Total problems: {len(dataset)}")
    print(f"Total responses: {sum(len(entry['responses']) for entry in dataset)}")
    
    return True


if __name__ == "__main__":
    # Test the dataset generation
    dataset = generate_synthetic_dataset()
    validate_dataset_diversity(dataset)
    
    # Show a sample entry
    print("\nSample dataset entry:")
    print(json.dumps(dataset[0], indent=2))
