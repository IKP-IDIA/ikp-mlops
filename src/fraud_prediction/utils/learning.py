import os 

def setup_logging_dir(dirname:str, file_name:str)-> str:
    for path in [dirname]:
        if not os.path.exists(path):
            os.makedirs(path)
        return os.path.join(dirname,file_name)
        logger.info(f"Directory and file path set up at: {os.path.join(dirname,file_name)} ")
        
            