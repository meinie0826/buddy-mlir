import numpy as np

def spmm_csr(values, col_indices, row_ptr, dense):
    num_rows = len(row_ptr) - 1
    num_cols = dense.shape[1]
    result = np.zeros((num_rows, num_cols))
    
    for i in range(num_rows):
        row_start = row_ptr[i]
        row_end = row_ptr[i+1]
        
        for k in range(row_start, row_end):
            col = col_indices[k]
            val = values[k]
            
            for j in range(num_cols):
                result[i, j] += val * dense[col, j]
    
    return result

# test data
values = np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32)
col_indices = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
row_ptr = np.array([0, 2, 4, 6, 8], dtype=np.int32)

# dense matrix (4x3)
dense = np.full((4, 3), 3.0, dtype=np.float32)

# execute spmm
result = spmm_csr(values, col_indices, row_ptr, dense)

print("Result matrix:")
print(result)