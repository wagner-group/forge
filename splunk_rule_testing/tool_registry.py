import traceback
from typing import Dict, Any, Optional
from detection import Colors

class ToolRegistry:
    def __init__(self, engine=None, target_log: str = "", gt_index: str = "", main_dataset: str = ""):
        self.engine = engine
        self.target_log = target_log
        self.gt_index = gt_index
        self.main_dataset = main_dataset
        self.finalized_rule: Optional[str] = None
        self.is_done = False
        self.test_passed = False  # Ensures test_rule must pass before submit_rule is accepted
        self.filter_dataset_called = False # Ensures filter_dataset must be called before submit_rule
        self.last_tested_rule: Optional[str] = None # Stores the rule from the last successful test_rule call
        self._failed_rules: set = set()  # Tracks previously-failed rules to detect loops
        self._transient_datasets: set = set() # Track datasets created during the session

    def test_rule(self, thought: str, rule: str) -> Dict[str, Any]:
        """
        Test a candidate detection rule against the suspicious input log.
        You MUST use this tool to verify your logic before submitting!
        Check the success field in the returned response to see if it worked.

        Args:
            thought: A brief explanation of your analysis and strategy for this test.
            rule: The detection rule code (Python, Splunk SPL, or Regex depending on the mode).
        """
        try:
            if not self.engine:
                return {"success": False, "message": "No detection engine is available in the current context.", "matches": 0}
            
            # test_pattern handles running against gt_index or the memory store
            success, msg, count = self.engine.test_pattern(rule, self.gt_index)

            if not (success and count > 0) and rule in self._failed_rules:
                return {
                    "success": False,
                    "message": (
                        "CRITICAL: You have already tested this exact rule and it failed. "
                        "Do NOT retry it. Take a completely different approach — "
                        "switch from regex to literal `in` checks, or vice versa."
                    ),
                    "matches": 0
                }
            if not (success and count > 0):
                self._failed_rules.add(rule)

            if success and count > 0:
                self.test_passed = True
                self.last_tested_rule = rule # Save for parameterless submit_rule(). Only updated on success.
                return {
                    "success": True, 
                    "message": "Passed! The rule successfully matched the input log without syntax errors. You can now use submit_rule() to finalize, or you can continue to refine the rule and call test_rule() again to verify improvements.", 
                    "matches": count
                }
            elif success and count == 0:
                return {
                    "success": False,
                    "message": "Rule is syntactically valid but it did NOT match the input log. Adjust your rule to be broader.",
                    "matches": 0
                }
            else:
                return {
                    "success": False,
                    "message": f"Syntax Error or Matching Failure: {msg}",
                    "matches": 0
                }
        except Exception as e:
            return {"success": False, "message": f"Runtime Exception during testing: {traceback.format_exc()}", "matches": 0}

    def submit_rule(self, thought: str, final_rule: Optional[str] = None) -> Dict[str, Any]:
        """
        Submit your finalized, tested detection rule. 
        You can call this without arguments to submit the rule that most recently passed test_rule().
        
        Args:
            thought: A final summary of why this rule is ready for submission.
            final_rule: Optional. The rule code to submit. If omitted, uses the last successful test.
        """
        if not (self.test_passed and self.filter_dataset_called):
            missing = []
            if not self.test_passed: missing.append("test_rule() (with success=True)")
            if not self.filter_dataset_called: missing.append("filter_dataset()")
            return {
                "success": False,
                "message": f"CRITICAL ERROR: You are missing required validations: {', '.join(missing)}. You are FORBIDDEN from calling submit_rule() until you have both successfully verified your code with test_rule() AND used filter_dataset() to evaluate false positives/negatives."
            }
        
        # Use the provided rule, or the last tested one if None
        rule_to_submit = final_rule if final_rule is not None else self.last_tested_rule
        
        if rule_to_submit is None:
            return {"success": False, "message": "Error: No rule was provided and no previous successful test was found."}

        self.finalized_rule = rule_to_submit
        self.is_done = True
        return {"success": True, "message": "Rule successfully submitted. Thank you."}

    def filter_dataset(self, thought: str, rule: str, dataset: str, new_dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply a detection rule to an existing dataset to find matching logs. 
        Use this to discover other instances of the behavior in the background "main" dataset, 
        verify your rule's robustness, or find potential False Positives.
        Up to 10 random matches are returned to give you a sample of the coverage.
        Optionally save the results as a new transient dataset for further iterative testing.

        Args:
            thought: Your reasoning for filtering this dataset.
            rule: The detection rule (regex, SPL, or Python) to apply.
            dataset: The name of the dataset to filter. Use "main" for the primary log repository.
            new_dataset: Optional. Name for the new transient dataset to store the matches. 
                         Cannot be "main" or an existing dataset name.
        """
        import random
        try:
            if not self.engine:
                return {"success": False, "message": "No engine available."}

            # Restrict accessible datasets to "main" and any transient subsets the agent
            # has created in this session. Prevents probing of internal stores (e.g. the
            # global "tn" store) to harvest label/origin information.
            allowed = {"main"} | self._transient_datasets
            if dataset not in allowed:
                return {"success": False, "message": (
                    f"Dataset '{dataset}' is not accessible. "
                    f"Available: {sorted(allowed)}"
                )}

            actual_dataset = self.main_dataset if dataset == "main" else dataset

            if new_dataset:
                if new_dataset == "main":
                    return {"success": False, "message": "New dataset name cannot be 'main'."}
                if new_dataset in self.engine.list_stores():
                    return {"success": False, "message": f"A dataset named '{new_dataset}' already exists."}

            # Run the filter
            success, msg, count = self.engine.test_pattern(rule, actual_dataset)
            if not success:
                return {"success": False, "message": f"Filtering failed: {msg}"}

            # Get matches
            matches = self.engine.get_matches(rule, actual_dataset, limit=1000)
            
            if new_dataset:
                match_lines = [m['_raw'] for m in matches]
                self.engine.create_store(new_dataset)
                self.engine.ingest_lines(match_lines, new_dataset)
                self._transient_datasets.add(new_dataset)

            # Sample 10 random matches
            sample = random.sample(matches, min(len(matches), 10))
            
            self.filter_dataset_called = True
            
            return {
                "success": True,
                "message": f"Found {count} matches in '{dataset}'." + (f" Saved to '{new_dataset}'." if new_dataset else ""),
                "match_count": count,
                "sample_matches": sample
            }
        except Exception as e:
            import traceback
            return {"success": False, "message": f"Error in filter_dataset: {str(e)}", "traceback": traceback.format_exc()}

    def cleanup_transient_datasets(self):
        """Delete all transient datasets created during this session."""
        if not self.engine:
            return
        for ds in self._transient_datasets:
            self.engine.delete_store(ds)
        self._transient_datasets.clear()

    def get_tool_list(self):
        """Return the functions themselves so the GenAI SDK can build the schema automatically"""
        return [self.test_rule, self.submit_rule, self.filter_dataset]

    def get_allowed_tool_names(self):
        """Returns the list of tools the model is allowed to call based on current state."""
        # print(f"[DEBUG] get_allowed_tool_names called for id={id(self)}. test_passed={self.test_passed}")
        allowed = ["test_rule", "filter_dataset"]
        if self.test_passed and self.filter_dataset_called:
            allowed.append("submit_rule")
        return allowed

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Manually map string calls from Model to local functions."""
        if name == "test_rule":
            return self.test_rule(**args)
        elif name == "submit_rule":
            return self.submit_rule(**args)
        elif name == "filter_dataset":
            return self.filter_dataset(**args)
        else:
            return {"success": False, "message": f"Unknown tool: {name}", "matches": 0}
