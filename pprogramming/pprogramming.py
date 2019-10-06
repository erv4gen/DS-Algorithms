import multiprocessing as mp
import numpy as np
import pandas as pd
def parallel_pandas(data,func):
        #fetching the number of cores in the VM  ( 8 cores in our case )
        cores = mp.cpu_count()
        #create a multiprocessing pool of n-cores
        pool = mp.Pool(cores)
        
        #split pandas dataframe into n chuncks 
        data_split = np.array_split(data,cores)
        
        #apply function to eack chunck in parrallel 
        data_res = pd.concat(pool.map(func,data_split))
        pool.close()
        pool.join()
        return data_res