import numpy as np
import os
import tempfile
class ExperimentLogger:

    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.logs = {}

    def log(self, name, value, step=None):

        if name not in self.logs:
            self.logs[name] = {"steps": [], "values": []}

        self.logs[name]["values"].append(value)

        if step is not None:
            self.logs[name]["steps"].append(step)
    
    def log_many(self, step=None, **kwargs):

        for key, value in kwargs.items():
            self.log(key, value, step=step)
    
    def get(self, name):
        return self.logs[name]

    def last(self, name):
        return self.logs[name]["values"][-1]
    
    def metric_at_switch(run, metric_name, alpha, N):
        P = alpha * N
    
        metric = run["logs"][metric_name]
    
        steps = metric["steps"]
        values = metric["values"]
    
        # exact match
        idx = np.where(steps == P)[0]
    
        if len(idx) == 0:
            return None  # or raise error
    
        return values[idx[0]]
    
    def to_dict(self):
        logs = {}

        for k, v in self.logs.items():
            logs[k] = {
            "steps": np.array(v["steps"]),
            "values": np.array(v["values"], dtype=object)
            }

        return {"metadata": self.metadata, "logs": logs}
        
    #Atomic save to not create bugs when ctrl+C during saving
    @staticmethod
    def append_to_file(path, logger):
        run = logger.to_dict()
    
        try:
            data = np.load(path, allow_pickle=True).item()
            runs = data.get("runs", [])
        except (FileNotFoundError, EOFError):
            runs = []
    
        runs.append(run)
    
        directory = os.path.dirname(path) or "."
        with tempfile.NamedTemporaryFile(delete=False, dir=directory, suffix=".npy") as tmp:
            tmp_path = tmp.name
    
        np.save(tmp_path, {"runs": runs}, allow_pickle=True)
        os.replace(tmp_path, path)


    @staticmethod
    def load_group(path):

        if not os.path.exists(path):
            return []

        data = np.load(path, allow_pickle=True).item()
        return data.get("runs", [])
    
    @staticmethod   
    def delete_runs(path, predicate):   
        
        if not os.path.exists(path):    
            print("File does not exist.")   
            return  
        
        data = np.load(path, allow_pickle=True).item()  
        runs = data.get("runs", []) 
        
        initial_len = len(runs) 
        
        filtered_runs = [   
            r for r in runs 
            if not predicate(r["metadata"]) 
        ]   
        
        deleted = initial_len - len(filtered_runs)  
        
        np.save(path, {"runs": filtered_runs}, allow_pickle=True)   
        
        print(f"Deleted {deleted} runs.")   