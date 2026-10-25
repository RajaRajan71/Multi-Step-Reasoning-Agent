import os
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

# Load environment variables from a .env file
load_dotenv()

# Initialize Gemini Client
client = genai.Client()
MODEL_ID = "gemini-flash-latest"

# ==========================================
# 1. PYDANTIC SCHEMAS (Guaranteed Structured Output)
# ==========================================

class PlanResponse(BaseModel):
    steps: List[str]

class ExecutorResponse(BaseModel):
    intermediate_work: str
    final_answer: str

class CheckItem(BaseModel):
    check_name: str
    passed: bool
    details: str

class VerifierResponse(BaseModel):
    checks: List[CheckItem]
    is_overall_passed: bool
    feedback: str

# ==========================================
# 2. PROMPTS DEFINITION (With Few-Shot Guidance)
# ==========================================

PLANNER_PROMPT = """You are an expert analytical planning agent. 
Given a user question, break it down into a concise, numbered step-by-step plan.
Example: For calculating a discount, steps should include: 1. Identify original price and discount percentage, 2. Compute discount amount, 3. Subtract from original price.

Question: {question}
"""

EXECUTOR_PROMPT = """You are an execution agent. Follow the provided step-by-step plan to solve the question. 
Show your intermediate calculations and deductions clearly.

Question: {question}
Plan: {plan}
"""

VERIFIER_PROMPT = """You are a strict verification agent. Review the original question and the proposed solution.
Check for logical consistency, arithmetic accuracy, constraint compliance, and unit correctness.

Question: {question}
Proposed Solution: {solution}
"""

# ==========================================
# 3. AGENT CORE FUNCTIONS
# ==========================================

def call_structured_llm(prompt: str, schema_model: BaseModel) -> dict:
    """Calls Gemini with strict Pydantic schema enforcement."""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_model,
            temperature=0.1
        ),
    )
    return json.loads(response.text)

def run_agent(question: str, max_retries: int = 2) -> Dict[str, Any]:
    retries = 0
    plan_text = ""
    checks_log = []
    status = "failed"
    final_answer = ""
    reasoning_visible = ""

    while retries <= max_retries:
        try:
            # Phase 1: Planner
            planner_input = PLANNER_PROMPT.format(question=question)
            plan_res = call_structured_llm(planner_input, PlanResponse)
            plan_steps = plan_res.get("steps", [])
            plan_text = " -> ".join(plan_steps)

            # Phase 2: Executor
            executor_input = EXECUTOR_PROMPT.format(question=question, plan=json.dumps(plan_steps))
            exec_res = call_structured_llm(executor_input, ExecutorResponse)
            intermediate_work = exec_res.get("intermediate_work", "")
            final_answer = exec_res.get("final_answer", "")

            # Phase 3: Verifier
            verifier_input = VERIFIER_PROMPT.format(
                question=question, 
                solution=f"Working: {intermediate_work} | Answer: {final_answer}"
            )
            ver_res = call_structured_llm(verifier_input, VerifierResponse)
            checks_log = ver_res.get("checks", [])
            is_passed = ver_res.get("is_overall_passed", False)

            reasoning_visible = f"Solved via {len(plan_steps)} planned steps with verified calculations."

            if is_passed:
                status = "success"
                break
            else:
                retries += 1
                reasoning_visible = f"Retry {retries}: Previous attempt failed verification due to: {ver_res.get('feedback', 'Unknown error')}."
        except Exception as e:
            retries += 1
            checks_log.append({"check_name": "Exception Handling", "passed": False, "details": str(e)})

    # Final Output Schema Compliance
    return {
        "answer": final_answer if status == "success" else "Unable to verify correct answer.",
        "status": status,
        "reasoning_visible_to_user": reasoning_visible,
        "metadata": {
            "plan": plan_text,
            "checks": checks_log,
            "retries": retries if status == "success" else max_retries
        }
    }

# ==========================================
# 4. TEST SUITE & CLI INTERFACE
# ==========================================

TEST_QUESTIONS = [
    # Easy Questions (Arithmetic / Time / Totals)
    "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?",
    "Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?",
    "A book costs $15. If there is a 20% discount, what is the final price?",
    "What is 45 * 12 + 150?",
    "If today is Wednesday, what day of the week will it be in 10 days?",
    # Tricky / Multi-step / Boundary Questions
    "A meeting needs 60 minutes. There are free slots: 09:00-09:30, 09:45-10:30, 11:00-12:00. Which slots can fit the meeting?",
    "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If 5 machines take 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
]

def run_test_suite():
    print("=== Running Agent Test Suite ===")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Test {i}] Question: {q}")
        result = run_agent(q)
        print(json.dumps(result, indent=2))
        print("-" * 50)
        time.sleep(2)  # Pause to prevent socket exhaustion / rate-limiting

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_test_suite()
    else:
        print("Interactive Agent CLI (Type 'exit' to quit)")
        while True:
            user_q = input("\nEnter question: ")
            if user_q.lower() == 'exit':
                break
            res = run_agent(user_q)
            print(json.dumps(res, indent=2))