# Multi-Step Reasoning Agent with Self-Checking

A robust Python reasoning agent built with the Google GenAI SDK (`google-genai`). It implements an autonomous three-phase architecture (**Planner → Executor → Verifier**) featuring automated retry loops, strict Pydantic-backed structured outputs, and a complete test suite covering math, logic, time intervals, and edge cases.

---

## Project Structure

- `agent.py`: Core agent architecture, prompt definitions, Pydantic schemas, execution loop, CLI, and test suite.
- `.env`: Environment configuration file storing your `GEMINI_API_KEY`.
- `requirements.txt`: Required dependencies (`google-genai`, `python-dotenv`, `pydantic`).
- `run_logs.txt`: Captured output logs from the test suite execution.

---

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key:**
   Create a `.env` file in the project folder and add your key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Run the Test Suite:**
   ```bash
   python agent.py test
   ```

4. **Run the Interactive CLI:**
   ```bash
   python agent.py
   ```

---

## Architecture & Prompt Rationale

- **Decoupled Phases:** The agent cleanly separates responsibilities to ensure high accuracy. The Planner breaks down questions into sequential steps, the Executor computes intermediate results, and the Verifier performs independent multi-criteria checks (arithmetic, logic, constraints).
- **Pydantic Structured Outputs:** Utilizing Google GenAI’s `response_schema` with Pydantic models guarantees 100% JSON schema compliance, eliminating parsing errors.
- **Self-Correcting Loop:** If the verifier rejects a solution (`is_overall_passed: false`), the agent captures feedback and retries up to 2 times automatically.

---

## Challenges & Future Improvements

### Challenges Overcome
* **JSON Parsing Fragility:** Early iterations using manual parsing encountered JSON formatting issues. Integrating native SDK Pydantic schema validation completely resolved this.
* **Network Socket Timeouts:** Adding short delays (`time.sleep`) between test iterations solved Windows network socket timeout errors during rapid batch execution.

### Future Enhancements
* Add persistent database logging (SQLite) for audit trails.
* Integrate an isolated Python code execution sandbox for complex calculations.
