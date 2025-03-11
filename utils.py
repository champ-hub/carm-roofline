import os
import math

CONFIG_FILE = "./config/auto_config/config.txt"

def read_library_path(tag):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as file:
            for line in file:
                if line.strip() == "":
                    continue
                parts = line.strip().split("=")
                if len(parts) == 2:
                    key, value = parts
                    if key == tag:
                        return value
    return None

def write_library_path(tag, path):
    with open(CONFIG_FILE, "a") as file:
        file.write(f"{tag}={path}\n")

def make_power_of_two_ticks(min_val, max_val):
    #Ensure min_val > 0 to avoid log2 errors. If <=0, adjust logic as needed.
    min_val = max(min_val, 0.0000000001)
    max_val = max(max_val, 0.0000000001)
    start_exp = math.floor(math.log2(min_val))
    end_exp = math.ceil(math.log2(max_val))
    tickvals = [2**i for i in range(start_exp, end_exp+1)]
    ticktext = [f"2<sup>{i}</sup>" for i in range(start_exp, end_exp+1)]
    return tickvals, ticktext

def ensure_list(marker_dict, attr_name, default_value, n_points):
        #If marker[attr_name] doesn't exist or is not a list, convert it to a repeated list.
        if attr_name not in marker_dict:
            return [default_value] * n_points
        
        val = marker_dict[attr_name]
        if isinstance(val, list):
            return val
        else:
            return [val] * n_points