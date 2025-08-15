# Author: Mohamed Hamdy
# Email: m-hamdy-m@outlook.com
# Date: 15 July 2025
# Description: MLP model
# -----------------------------------

import pandas as pd 
import numpy as np
from scipy.stats import wilcoxon
from autorank import autorank, create_report

def wilcoxon_paried_test(results_df, alpha=0.05) : 
    methods = results_df.columns.values
    results = pd.DataFrame(data = np.zeros((len(methods) , len(methods))) , columns = methods , index= methods)
    p_values = pd.DataFrame(data = np.zeros((len(methods) , len(methods))) , columns = methods , index= methods)    
    for i , curr_method in enumerate(methods) : 
        for comp_method in methods : 
            if comp_method == curr_method : 
                results.loc[curr_method , comp_method ] = 0
                p_values.loc[curr_method , comp_method ] = 0
            else :
                _ , p_greater = wilcoxon(x =results_df[curr_method] , y = results_df[comp_method] , alternative="greater")
                _ , p_lower = wilcoxon(x =results_df[curr_method] , y = results_df[comp_method] , alternative="less")
                _ , p_diff = wilcoxon(x =results_df[curr_method] , y = results_df[comp_method] , alternative="two-sided")
                if p_greater < alpha : 
                    results.loc[curr_method , comp_method ] = 1
                    p_values.loc[curr_method , comp_method ] = p_greater

                elif p_lower <alpha : 
                    results.loc[curr_method , comp_method ] = -1
                    p_values.loc[curr_method , comp_method ] = p_lower
                else : 
                    results.loc[curr_method , comp_method ] = 0
                    p_values.loc[curr_method , comp_method ] = p_diff

    return results , p_values,  

def highlight_max(s):
    original = ["white"] * len(s)
    original[:5] = ["red", "orange", "yellow", "green", "blue"]
    original = original[:len(s)]
    sorted_indices = s.argsort()[::-1]
    colors = np.array(original.copy())
    colors[sorted_indices.values.astype(int)] = original
    return [f'color: {colors[i]}; font-weight: bold; font-size: 1.2em' for i in range(len(s))]
