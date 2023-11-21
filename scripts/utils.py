from pathlib import Path
import logging
import datetime
import sys
import yaml
import os


def load_config_and_check_arg(arguments):
    """Load configuration file and check input arguments"""

    # Loading configuration file
    try:
        with open(arguments.data_configure) as infile:
            load_configure = yaml.safe_load(infile)
    except Exception:
        sys.stderr.write("Please specify valid yaml file.")
        sys.exit(1)

    data_dir = Path(arguments.data_dir)
    work_dir = Path(arguments.work_dir)

    save_dir = work_dir.joinpath(load_configure['experiment_dir'], "results")
    genotype_dir = data_dir.joinpath(load_configure['genotype_data_path'])
    weight_tissue_dir = data_dir.joinpath(load_configure['weights_data_path'])
    dosage_prefix = arguments.dosage_prefix
    weight_prefix = arguments.weight_prefix
    dosage_end_prefix = arguments.dosage_end_prefix
    weight_end_prefix = arguments.weight_end_prefix
    weight_tissue = arguments.weight_tissue
    sample_data = data_dir.joinpath(load_configure['sample_data'])
    reference_data = data_dir.joinpath(load_configure['reference_data'])

    # Check all the arguments
    check_all_specified_arguments(genotype_dir=genotype_dir,
                                  weight_tissue_dir=weight_tissue_dir,
                                  dosage_prefix=dosage_prefix,
                                  weight_prefix=weight_prefix,
                                  dosage_end_prefix=dosage_end_prefix,
                                  weight_end_prefix=weight_end_prefix,
                                  weight_tissue=weight_tissue,
                                  sample_data=sample_data,
                                  reference_data=reference_data)

    # Update the configuration file
    with open(arguments.data_configure) as infile:
        load_configure = yaml.safe_load(infile)
        load_configure['save_dir'] = save_dir
        load_configure['genotype_dir'] = genotype_dir
        load_configure['weight_tissue_dir'] = weight_tissue_dir
        load_configure['sample_data'] = sample_data
        load_configure['reference_data'] = reference_data
        load_configure['dosage_prefix'] = dosage_prefix
        load_configure['weight_prefix'] = weight_prefix
        load_configure['dosage_end_prefix'] = dosage_end_prefix
        load_configure['weight_end_prefix'] = weight_end_prefix
        load_configure['weight_tissue'] = weight_tissue

        return load_configure


def check_all_specified_arguments(**arguments):
    """ Check all specified arguments for plausibility."""

    # check existence of genotype dir and genotype file
    if not arguments['genotype_dir'].exists():
        raise Exception("Specified genotype_dir " +
                        str(arguments['genotype_dir']) +
                        " does not exist. Please double-check.")
    else:
        genotype_file = [
            arguments['genotype_dir'] /
            (arguments['dosage_prefix'] + str(x) +
             arguments['dosage_end_prefix']) for x in range(1,23)
        ]

        if not check_exist_files(genotype_file):
            raise Exception("Specified genotype_file under" +
                            str(arguments['genotype_dir']) +
                            " does not exist. Please double-check.")

    # check existence weight dir and weight file
    if not arguments['weight_tissue_dir'].exists():
        raise Exception("Specified weight_dir " +
                        str(arguments['weight_tissue_dir']) +
                        " does not exist. Please double-check.")
    else:
        weight_file = [
            arguments['weight_tissue_dir'] /
            (arguments['weight_prefix'] + "_" + arguments['weight_tissue'] +
             "_imputed_europeans_tw_0.5_signif" +
             arguments['weight_end_prefix'])
        ]

        if not check_exist_files(weight_file):
            raise Exception(
                "Specified weight_file under genotype_dir does not exist. Please double-check."
            )
    # check existence sample file
    if not arguments['sample_data'].is_file():
        raise Exception("Specified sample_file " +
                        str(arguments['sample_data']) +
                        " does not exist in. Please double-check.")

    # check existence reference file
    if not arguments['reference_data'].is_file():
        raise Exception("Specified sample_file " +
                        str(arguments['reference_data']) +
                        " does not exist in. Please double-check.")


def check_exist_files(list_files):
    """Check if each file within a list exists."""

    if not isinstance(list_files, list):
        list_files = [list_files]

    check = True
    for file_to_check in list_files:
        if not Path(file_to_check).is_file():
            check = False
    return check

def check_exist_directories(list_dirs):
    """
    Check if each directory within a list exists
    """
    if not isinstance(list_dirs, list):
            list_dirs = [list_dirs]
    
    check = True
    for dir_to_check in list_dirs:
        if not Path(dir_to_check).exists():
           check = False
    return check

def logging_config(identifier):
    """Set up logging function."""

    # Create a unique filename using the current timestamp and identifier
    timestamp = datetime.datetime.now().today().isoformat()
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
    """Save model hyperparameters/metadata to output directory."""

    output_file = construct_filename(output_dir,
                                     file_descriptor,
                                     extension=".txt")

    if check_exist_files([output_file]):
        raise Exception("'Results file ", str(output_file), " exist alreadly.")
    else:
        with open(output_file, 'w') as f:
            f.write('+++++++++++ CONFIG INFORMATION +++++++++++')
            f.write('Summary Statistics:\n')
            f.write('## Weight databse:\t' + str(arguments['weights_mdoel']) +
                    '\n')
            f.write('+++++++++++ CONFIG INFORMATION +++++++++++')