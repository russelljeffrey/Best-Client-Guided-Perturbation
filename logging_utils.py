import pandas as pd
import os

class Logger:
    def __init__(self, log_file="FL_log.csv", scenario="normal"):
        self.log_file = log_file
        self.scenario = scenario  # Store the scenario type ("normal" or "attack")
        
        # Initialize the log file with scenario-specific headers
        if not os.path.exists(self.log_file):
            if self.scenario == "normal":
                columns = ["Round_Accuracy", "Total_Communication_Cost", "Wall_Clock_Time"]
            elif self.scenario == "attack":
                columns = ["Round_Test_Error", "Round_Test_Accuracy", "Wall_Clock_Time"]
            else:
                raise ValueError(f"Unsupported scenario: {self.scenario}")
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.log_file, index=False)

    def log_round(self, round_accuracy, total_communication_cost, wall_clock_time):
        """Log the metrics for a single round to the CSV file."""
        # Create a new row with the updated metrics
        new_row = {
            "Round_Accuracy": round_accuracy,
            "Total_Communication_Cost": total_communication_cost,
            "Wall_Clock_Time": wall_clock_time
        }
        # Append the row to the CSV file
        df = pd.DataFrame([new_row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

    def log_attack_round(self, avg_test_error, avg_test_accuracy, wall_clock_time):
        """Log the metrics for a single round in attack scenarios."""
        # Create a new row with the metrics
        new_row = {
            "Round_Test_Error": avg_test_error,
            "Round_Test_Accuracy": avg_test_accuracy,
            "Wall_Clock_Time": wall_clock_time
        }
        # Append the row to the CSV file
        df = pd.DataFrame([new_row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)