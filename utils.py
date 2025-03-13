import os
import math
from decimal import Decimal

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

def custom_round(value, digits=4):
    if value == 0:
        return 0  #Directly return 0 if the value is 0
    elif abs(value) >= 1:
        #For numbers greater than or equal to 1, round normally
        return round(value, digits)
    else:
        #For numbers less than 1, find the position of the first non-zero digit after the decimal
        dec_val = Decimal(str(value))
        str_val = format(dec_val, 'f')
        if 'e' in str_val or 'E' in str_val:  #Check for scientific notation
            return round(value, digits)
        
        #Count positions until first non-zero digit after the decimal
        decimal_part = str_val.split('.')[1]
        leading_zeros = 0
        for char in decimal_part:
            if char == '0':
                leading_zeros += 1
            else:
                break
        
        #Adjust the number of digits based on the position of the first significant digit
        total_digits = digits + leading_zeros
        return round(value, total_digits)