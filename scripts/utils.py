from pathlib import Path
import logging
import datetime
import os



def check_exist_files(list_files):
    """
    Check if each file within a list exists.
    """

    if not isinstance(list_files, list):
        list_files = [list_files]

    check = True
    for file_to_check in list_files:
        if not Path(file_to_check).is_file():
            check = False
    return check

def check_exist_directories(list_dirs):
    """
    Check if each directory within a list exists.
    """
    if not isinstance(list_dirs, list):
            list_dirs = [list_dirs]
    
    check = True
    for dir_to_check in list_dirs:
        if not Path(dir_to_check).exists():
           check = False
    return check

def logging_config(identifier, timestamp=datetime.datetime.now().today().isoformat()):
    """
    Set up logging function.
    """

    # Create a unique filename using the current timestamp and identifier
    timestamp = timestamp + "_" + identifier
    filename = f"logfile_{timestamp}.log"
    current_path = os.getcwd()

    # Configure the logger to write to the file
    file_handler = logging.FileHandler(
        os.path.join(current_path, "Log", filename))
    logger = logging.getLogger(filename)

    # Set up the logger
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                        handlers=[file_handler])

    return logger


def construct_filename(output_dir,
                       file_descriptor,
                       extension,
                       *args,
                       **kwargs):
    """
    Construct a filename for varying experimental parameters.
    
    For example:
    >>> construct_filename('output_dir', 'output', '.tsv',
    ...                    'expression', 'signal',
    ...                    s=42, n=5000)
    output_dir/expression_signal_s42_n5000_output.tsv
    """

    if len(args) == 0 and len(kwargs) == 0:
        return Path(output_dir, 
                    '{}{}'.format(file_descriptor, extension))
    
    elif len(args) == 0:
        return Path(output_dir, 
                    '{}_{}{}'.format('_'.join(f'{k}{v}' for k, v in kwargs.items() 
                                                            if v is not None), 
                                    file_descriptor, 
                                    extension))
    
    elif len(kwargs) == 0:
        return Path(output_dir,
                    '{}_{}{}'.format('_'.join(ar for ar in args 
                                              if ar is not None),
                                    file_descriptor, 
                                    extension))
    else:
        return Path(output_dir, 
                    '{}_{}_{}{}'.format('_'.join(ar for ar in args if ar is not None),
                                        '_'.join(f'{k}{v}' for k, v in kwargs.items()
                                                               if v is not None), 
                                        file_descriptor, 
                                        extension))


def write_configure_options(arguments,
                            output_dir,
                            file_descriptor):
    """
    Save model hyperparameters/metadata to output directory.
    """

    output_file = construct_filename(output_dir,
                                     file_descriptor,
                                     extension=".txt")

    if check_exist_files([output_file]):
        raise Exception("'Results file ", str(output_file), " exist alreadly.")
    else:
        with open(output_file, 'w') as f:
            f.write('+++++++++++ CONFIG INFORMATION +++++++++++')
            f.write('Summary Statistics:\n')
            f.write('## Weight databse:\t' + str(arguments['weights_mdoel']) +'\n')
            f.write('+++++++++++ CONFIG INFORMATION +++++++++++')          