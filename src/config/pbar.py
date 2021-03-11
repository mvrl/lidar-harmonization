from tqdm import tqdm

def get_pbar(
        iter, 
        total, 
        desc, 
        position, 
        print_ratio=.25,  # change to 1 for real time progress updates in term
        disable=False, 
        dynamic_ncols=True, 
        leave=False):    

    format_desc = "  "*position+f"{desc}"
    min_iters = total//(1/print_ratio)
    pbar = tqdm(iter, desc=format_desc, total=total, position=position,     
                    dynamic_ncols=dynamic_ncols, disable=disable, leave=leave,
                    miniters=min_iters
                )    
    
    return pbar    
