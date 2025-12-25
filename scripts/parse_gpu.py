
import sys

def parse_bhosts(filepath):
    gpu_counts = {}
    free_gpus = 0
    total_gpus = 0
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    start_idx = 1
    
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) < 8:
            continue
            
        # identifying columns based on standard bhosts -gpu output
        # HOST_NAME GPU_ID MODEL MUSED MRSV NJOBS RUN SUSP RSV
        # Sometimes HOST_NAME is empty if it's the same host as previous line
        
        # We need to find the MODEL column. 
        # It usually contains "NVIDIA"
        model = None
        njobs_idx = -1
        
        for i, part in enumerate(parts):
            if "NVIDIA" in part:
                model = part
                # NJOBS is typically 3 columns after MODEL (MUSED, MRSV, NJOBS)
                # But MUSED and MRSV can have units like 10.7G, 0M
                # Let's count from the end or finding the model index
                njobs_idx = i + 3
                break
        
        if model:
            total_gpus += 1
            if model not in gpu_counts:
                gpu_counts[model] = {'total': 0, 'free': 0}
            
            gpu_counts[model]['total'] += 1
            
            # Check NJOBS
            # If parts has enough columns
            if njobs_idx < len(parts):
                njobs = parts[njobs_idx]
                try:
                    if int(njobs) == 0:
                        gpu_counts[model]['free'] += 1
                        free_gpus += 1
                except ValueError:
                    pass

    print(f"Total GPUs: {total_gpus}")
    print(f"Free GPUs: {free_gpus}")
    print("By Model:")
    for model, counts in gpu_counts.items():
        print(f"  {model}: Total {counts['total']}, Free {counts['free']}")

if __name__ == "__main__":
    parse_bhosts('/home/projects/dharel/nadavt/.cursor/projects/home-projects-dharel-nadavt-repos-newt/agent-tools/e2f24237-2deb-461a-8e96-9244151f0724.txt')
