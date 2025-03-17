import csv
import os
import math
import plotly.graph_objects as go
import numpy as np

import utils as ut

def read_csv_file(file_path):
    data_list = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        machine_name = header[1]
        l1_size = int(header[3])
        l2_size = int(header[5])
        l3_size = int(header[7])

        header2 = next(reader)
        for row in reader:
            if not row or not ''.join(row).strip():
                continue
            data = {}
            data['Date'] = row[0]
            data['ISA'] = row[1]
            data['Precision'] = row[2]
            data['Threads'] = int(row[3])
            data['Loads'] = int(row[4])
            data['Stores'] = int(row[5])
            data['Interleaved'] = row[6]
            data['DRAMBytes'] = int(row[7])
            data['FPInst'] = row[8]
            data['L1'] = float(row[9])
            data['L2'] = float(row[11])
            data['L3'] = float(row[13])
            data['DRAM'] = float(row[15])
            data['FP'] = float(row[17])
            data['FP_FMA'] = float(row[19])
            data_list.append(data)

    return machine_name, l1_size, l2_size, l3_size, data_list

def read_application_csv_file(file_path):
    if not os.path.exists(file_path):
        print("Application file does not exist:", file_path)
        return False

    data_list = []
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)

            if header is None:
                print("File is empty:", file_path)
                return False
            
            for row in reader:
                if row:
                    data = {
                        'Date': row[0],
                        'Method': row[1],
                        'Name': row[2],
                        'ISA': row[3],
                        'Precision': row[4],
                        'Threads': row[5],
                        'AI': float(row[6]),
                        'GFLOPS': float(row[7]),
                        'Bandwidth': float(row[8]),
                        'Time': float(row[9])
                    }
                    data_list.append(data)

    except Exception as e:
        print("Failed to read the file:", file_path, "Error:", e)
        return False
    return data_list if data_list else False


def extract_last_segment(s):
    return s.split("_")[-1] if "_" in s else s

def extract_prefix(s):
    if "_" in s:
        return s.rsplit("_", 1)[0]
    return s

def interpolate_color(start_color, end_color, factor):
    r = int(start_color[0] + factor * (end_color[0] - start_color[0]))
    g = int(start_color[1] + factor * (end_color[1] - start_color[1]))
    b = int(start_color[2] + factor * (end_color[2] - start_color[2]))
    return f'rgb({r},{g},{b})'

def construct_query(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date):
    query_parts = []
    if ISA:
        query_parts.append(f"ISA == '{ISA}'")
    if Precision:
        query_parts.append(f"Precision == '{Precision}'")
    if Threads:
        query_parts.append(f"Threads == {Threads}")
    if Loads:
        query_parts.append(f"Loads == {Loads}")
    if Stores:
        query_parts.append(f"Stores == {Stores}")
    if Interleaved:
        query_parts.append(f"Interleaved == '{Interleaved}'")
    if DRAMBytes:
        query_parts.append(f"DRAMBytes == {DRAMBytes}")
    if FPInst:
        query_parts.append(f"FPInst == '{FPInst}'")
    if Date:
        query_parts.append(f"Date == '{Date}'")

    return " and ".join(query_parts) if query_parts else None

def calculate_roofline(values, min_ai):
    aidots = [0]*3
    FPaidots = [0]*2
    gflopdots = [0]*3
    FPgflopdots = [0]*2

    ai = np.linspace(min(0.00390625,min_ai), 256, num=200000)
    traces = []
    cache_levels = ['L1', 'L2', 'L3', 'DRAM']
    
    dots = {}
    if values[5] > 0:
        peak_flops = values[5]
    else:
        peak_flops = values[4]

    for cache_level in cache_levels:
        if values[cache_levels.index(cache_level)] > 0:
            aidots = [0, 0, 0]
            gflopdots = [0, 0, 0]

            y_values = ut.carm_eq(ai, values[cache_levels.index(cache_level)], peak_flops)
            y_special = ut.carm_eq(0.00390625, values[cache_levels.index(cache_level)], peak_flops)

            #Find the point where y_values stops increasing or reaches a plateau
            for i in range(1, len(y_values)):
                if y_values[i - 1] == y_values[i]:
                    aidots[1] = float(ai[i - 1])
                    break
            else:
                #If no break occurred in the loop
                aidots[1] = float(ai[-1])
                i = len(y_values) - 1

            mid_ai = np.sqrt(aidots[1]*min(0.00390625,min_ai))
            mid_gflops = np.sqrt(y_values[0]*y_values[i - 1])

            dots[cache_level] = {
                "start": [min(0.00390625,min_ai), y_values[0]],
                "mid": [mid_ai, mid_gflops],
                "ridge": [aidots[1], y_values[i - 1]],
                "end": [ai[-1], y_values[-1]]
            }

    for i in range(4):
        if values[i]:
            top_roof = values[i]
            break

    y_values = ut.carm_eq(ai, top_roof, values[4])

    for i in range(1, len(y_values)):
        if(y_values[i-1] == y_values[i]):
            FPaidots[0] = float(ai[i-1])
            break
    FPgflopdots[0]= y_values[i-1]

    FPaidots[1] = ai[199999]
    FPgflopdots[1] = y_values[199999]

    dots[values[6]] = {
                "ridge": [FPaidots[0], FPgflopdots[0]],
                "end": [FPaidots[1], FPgflopdots[1]]
            }

    return dots

def plot_roofline(values, dots, name_suffix, ISA, line_legend, line_size, line_legend_detailed):
    import numpy as np
    aidots = [0]*3
    FPaidots = [0]*2
    gflopdots = [0]*3
    FPgflopdots = [0]*2

    ai = np.linspace(0.00390625, 256, num=200000)
    traces = []
    cache_levels = ['L1', 'L2', 'L3', 'DRAM']
    if name_suffix == "":
        colors = ['black', 'black', 'black', 'black']
        color_inst = 'black'
    else:
        colors = ['red', 'red', 'red', 'red']
        color_inst = 'red'
    linestyles = ['solid', 'solid', 'dash', 'dot']

    for cache_level, color, linestyle in zip(cache_levels, colors, linestyles):
        
        cache_dots = dots.get(cache_level)
        if cache_dots:
            if line_legend_detailed:
                legend_text = f'{cache_level} {ISA.upper()} Bandwidth: {values[cache_levels.index(cache_level)]} GB/s'
            else:
                legend_text = f'{cache_level} {ISA.upper()}'
            aidots = [
                cache_dots["start"][0],
                cache_dots["ridge"][0],
                cache_dots["end"][0]
            ]
            gflopdots = [
                cache_dots["start"][1],
                cache_dots["ridge"][1],
                cache_dots["end"][1]
            ]
            trace = go.Scatter(
                x=aidots, y=gflopdots,
                mode='lines',
                text=['',f'{cache_level} {ISA.upper()} Peak Bandwidth: {values[cache_levels.index(cache_level)]} GB/s',f'FP FMA {ISA.upper()} Peak: {values[5]} GFLOP/s'],
                hovertemplate='<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>',
                line=dict(color=color, dash=linestyle, width=line_size),
                name=legend_text,
                showlegend=line_legend,
            )
            traces.append(trace)
    if values[4] > 0:
        aidots = [
            dots[values[6]]["ridge"][0],
            dots[values[6]]["end"][0]
        ]
        gflopdots = [
            dots[values[6]]["ridge"][1],
            dots[values[6]]["end"][1]
        ]
        if line_legend_detailed:
            legend_text = f'FP {values[6].upper()} {ISA.upper()}  Peak: {values[4]} GFLOP/s'
        else:
            legend_text = f'{values[6].upper()} {ISA.upper()}'
        if values[5] == 0:
            linedash = 'solid'
        else:
            linedash = 'dashdot'
        trace_inst = go.Scatter(
            x=aidots, y=gflopdots,
            mode='lines',
            text=[f'FP {ISA.upper()} {values[6].upper()} Peak Performance: {values[4]} GFLOP/s',f'FP {ISA.upper()} {values[6].upper()} Peak: {values[4]} GFLOP/s'],
            hovertemplate='<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>',
            line=dict(color=color_inst, dash=linedash, width=line_size),
            name=legend_text,
            showlegend=line_legend,
        )
        traces.append(trace_inst)
    if values[5] > 0:
        aidots = [
            dots["L1"]["ridge"][0],
            dots["L1"]["end"][0]
        ]
        gflopdots = [
            dots["L1"]["ridge"][1],
            dots["L1"]["end"][1]
        ]

        if line_legend_detailed:
            legend_text = f'FP FMA {ISA.upper()} Peak: {values[5]} GFLOP/s'
        else:
            legend_text = f'FMA {ISA.upper()}'

        trace_inst = go.Scatter(
            x=aidots, y=gflopdots,
            mode='lines',
            text=[f'FP {ISA.upper()} FMA Peak Performance: {values[5]} GFLOP/s',f'FP {ISA.upper()} FMA Peak: {values[5]} GFLOP/s'],
            hovertemplate='<b>%{text}</b><br>(%{x}, %{y})<br><extra></extra>',
            line=dict(color=color_inst, dash="solid", width=line_size),
            name=legend_text,
            showlegend=line_legend,
        )
        traces.append(trace_inst)
    
    return traces

def draw_annotation(values, lines, name_suffix, ISA, cache_level, graph_width, graph_height, anon_size, x_range=None, y_range=None):
    aidots = [0]*3
    gflopdots = [0]*3
    annotation = {}
    cache_levels = ['L1', 'L2', 'L3', 'DRAM']
    angle_degrees = {}

    if cache_level in cache_levels:
        if cache_level in lines and lines[cache_level]['ridge'][0] > 0:

            log_x1, log_x2 = math.log10(lines[cache_level]['start'][0]), math.log10(lines[cache_level]['ridge'][0])
            log_y1, log_y2 = math.log10(lines[cache_level]['start'][1]), math.log10(lines[cache_level]['ridge'][1])

            log_xmin, log_xmax = x_range[0], x_range[1]
            log_ymin, log_ymax = y_range[0], y_range[1]

            #Compute pixel coordinates based on log scale
            x1_pixel = ( (log_x1 - log_xmin) / (log_xmax - log_xmin) ) * graph_width
            x2_pixel = ( (log_x2 - log_xmin) / (log_xmax - log_xmin) ) * graph_width

            y1_pixel = graph_height - ( (log_y1 - log_ymin) / (log_ymax - log_ymin) ) * graph_height
            y2_pixel = graph_height - ( (log_y2 - log_ymin) / (log_ymax - log_ymin) ) * graph_height

            #Pixel slope
            pixel_slope = (y2_pixel - y1_pixel) / (x2_pixel - x1_pixel)

            #Convert pixel slope to angle in degrees
            angle_degrees[cache_level] = math.degrees(math.atan(pixel_slope))
    
    ai = np.linspace(0.00390625, 256, num=200000)
    traces = []
    
    if name_suffix == "1":
        colors = ['black', 'black', 'black', 'black']
        color_inst = 'black'
        factor = 1.3
    else:
        colors = ['red', 'red', 'red', 'red']
        color_inst = 'red'
        factor = 0.7
    linestyles = ['solid', 'solid', 'dash', 'dot']


    if cache_level in cache_levels and values[cache_levels.index(cache_level)] > 0:
        if cache_level in lines:
            aidots[0] = 0.00390625
            y_values = ut.carm_eq(ai, values[cache_levels.index(cache_level)], values[5])
            gflopdots[0]= y_values[0]
            for i in range(1, len(y_values)):
                if(y_values[i-1] == y_values[i]):
                    aidots[1] = float(ai[i-1])
                    break
            gflopdots[1]= y_values[i-1]

            annotation = go.layout.Annotation(
            x=math.log10(lines[cache_level]['mid'][0]*factor),
            y=math.log10(lines[cache_level]['mid'][1]*factor),
            text=f'{cache_level} {ISA} Bandwidth: {values[cache_levels.index(cache_level)]} GB/s',
            showarrow=False,
            font=dict(
                color=colors[0],
                size=anon_size,
            ),
            align="center",
            bgcolor="white",
            bordercolor=colors[0],
            borderwidth=1,
            textangle=angle_degrees[cache_level],
            name=f"{cache_level}_{name_suffix}"
            )
    if cache_level == "FMA" and values[5] > 0:
        mid_ai = np.sqrt(lines["L1"]['ridge'][0]*lines["L1"]['end'][0])
        mid_gflops = lines["L1"]['ridge'][1]
        annotation = go.layout.Annotation(
            x=math.log10(mid_ai),
            y=math.log10(mid_gflops),
            text=f'FP FMA {ISA} Peak: {values[5]} GFLOP/s',
            showarrow=False,
            font=dict(
                color=colors[0],
                size=anon_size,
            ),
            align="center",
            bgcolor="white",
            bordercolor=colors[0],
            borderwidth=1,
            textangle=0,
            name=f"FP_FMA_{name_suffix}"
            )
        
    if cache_level == "FP" and values[4] > 0:
        mid_ai = np.sqrt(lines["L1"]['ridge'][0]*lines["L1"]['end'][0])
        mid_gflops = values[4]
        annotation = go.layout.Annotation(
            x=math.log10(mid_ai),
            y=math.log10(mid_gflops),
            text=f'FP {values[6].upper()} {ISA} Peak: {values[4]} GFLOP/s',
            showarrow=False,
            font=dict(
                color=colors[0],
                size=anon_size,
            ),
            align="center",
            bgcolor="white",
            bordercolor=colors[0],
            borderwidth=1,
            textangle=0,
            name=f"FP_{name_suffix}"
            )  
    return annotation