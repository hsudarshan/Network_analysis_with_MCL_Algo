import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import normalize
from numpy import linalg as LA
import collections

def get_dict_str_2_int_mapping(ip):
    dict_nodes= {}
    for val in ip.values.flatten().tolist():
        if(val in dict_nodes):
            pass
        else:
            #if dict is empty - start with 0
            if(any(dict_nodes)==False):
                dict_nodes[val]=0
                pass
            else:
                #get max value of stored values
                key_with_max_value = max(dict_nodes, key=dict_nodes.get)
                #assign max_value + 1 to new key
                #store in dict and its new value
                dict_nodes[val]=dict_nodes[key_with_max_value]+1
                pass
    return dict_nodes

def ip_df_serialised(ip):
    abc = pd.Series(ip.dtypes).to_dict().values()
    for attr in abc:
        #if(attr==np.dtype('O')):
        #call function to get dict of mappings
        dict_mappings = get_dict_str_2_int_mapping(ip)
        #print("convert this to numberic")
        ip_new = ip.applymap(lambda x:dict_mappings[x])
        break
        #else:
        #pass
    return (ip_new,dict_mappings)

def populate_init_matrix(init_matrix,input_dataframe):
    """
    populate the given dummy init_matrix based on the input data recieved
    """
    for row in input_dataframe.itertuples():
        ls = list(row)
        init_matrix[ls[1],ls[2]] = 1
        init_matrix[ls[2],ls[1]] = 1

def normalize_matrix_column(matrix_tb_norm):
    """
    normalize the data column-wise, such that sum of the elements along a column is 1
    """
    return normalize(matrix_tb_norm, norm='l1', axis=0)

def expand_inflate_normalize(matrix_input,expansion_param,inflation_param,prune_by_rounding=False,rounding_precision=5):
    """
    core of the MCL algo-
    expand - stochastic matrix raise to the power of 'expansion_param'
    inflation - raising each element of the expanded matrix to 'inflation_param'
    normalization - normalize the expanded,inflated matrix to make it stochastic again
    prune_by_rounding - if True, performs a rounding-off to the 'rounding_precision' param(no.of decimal digits)
    """
    #expansion
    new_matrix = LA.matrix_power(matrix_input,expansion_param)
    #prune by rounding off to the nearest nth decimal point
    if(prune_by_rounding):
        new_matrix =np.around(new_matrix,decimals=rounding_precision)
    #inflation
    new_matrix =np.power(new_matrix,inflation_param)
    #print(new_matrix)
    #normalize
    new_norm_matrix = normalize_matrix_column(matrix_tb_norm=new_matrix)
    return new_norm_matrix

def check_matrix_di(matrix_prev,matrix_cur):
    """
    compare matrices in the current and prev cycle and determine if they have changed
    i.e., take difference of arrays, flatten them and sum up their magnitudes, if difference>0 then changed
    if no change, then check if all the non zero elements of columns are equal,if yes then stable matrix
    (doubly idempotent)
    """
    size = matrix_prev.shape[0]
    new_matrix = np.absolute(matrix_cur - matrix_prev)
    matrix_sum = np.sum(new_matrix)
    if(matrix_sum>0):
        return False
    else:
        #check if all elements column-wise are equal(non-zero)
        flag = True
        for i in range(0,matrix_cur.shape[0]):
            arr = matrix_cur[matrix_cur[:,i]>0,i]
            if(len(arr)>0):
                flag = flag and (arr == arr[0]).all()
        return flag

def extract_2_df_cluster_grouping(stable_matrix,exp_val,infl_val,result_df,min_positive_value=0):
    """
    from the stable matrix - identify the groups and set the cluster number accordingly in the result_df
    """
    num_rows = stable_matrix.shape[0]
    ref = np.arange(0,num_rows)
    cluster_list = []
    col_name = 'exp_'+str(exp_val)+'_infl_'+str(infl_val)
    cluster_counter = 1
    for i in range(0,num_rows-1):
        if(np.sum([np.greater(stable_matrix[i,:],min_positive_value)])>1):
            cluster_list=ref[np.greater(stable_matrix[i,:],min_positive_value)]
            #print(np.where([np.greater(new_norm_matrix[i,:],min_positive_value)]))
            result_df.loc[cluster_list,col_name]=cluster_counter
            cluster_counter+=1
    return cluster_list

import shutil
def write_result(input_file,result_df):
    folder_name = input_file[:-4]
    shutil.rmtree(folder_name, ignore_errors=True)
    os.mkdir(folder_name)
    os.chdir(folder_name)
    for param_combo in list(result_df.columns):
        #print(param_combo)
        output_file_name = param_combo+"_"+input_file[:-4]+".clu"
        #os.rmdir(input_file[:-4])
        with open(output_file_name,'w') as abc:
            start_str = '*Vertices '+str(len(result_df.index))+"\n"
            abc.write(start_str)
            x=result_df.loc[:,param_combo].values[:]
            np.savetxt(abc,x,delimiter=" ",fmt="%d")
    os.chdir("..")
