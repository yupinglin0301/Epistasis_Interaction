import numpy as np
import pandas as pd
import os
import re
import sqlite3
import sys
from tqdm import tqdm


class WeightsDB():

    def __init__(self,
                 beta_file):
        self.conn = sqlite3.connect(beta_file)

    def query(self,
              sql,
              args=None):

        c = self.conn.cursor()
        if args:
            for ret in c.execute(sql, args):
                yield ret
        else:
            for ret in c.execute(sql):
                yield ret


class GenotypeDataset():

    @staticmethod
    def get_all_dosages(genoytpe_dir,
                        dosage_prefix,
                        dosage_end_prefix,
                        unique_rsids,
                        reference_file,
                        logger):

        for chrfile in [
                x for x in sorted(genoytpe_dir.iterdir())
                if x.name.startswith(str(dosage_prefix))
                and x.name.endswith(str(dosage_end_prefix))
        ]:
            
            logger.info("Processing on {} ...".format(str(chrfile)))
            # get chr number
            chr_name = os.path.basename(chrfile).split(".")[0]
            chr_number = re.findall(r'\d+', chr_name)
            # get reference file with specific chr numner
            get_ref_chrfile = reference_file.loc[reference_file.CHR == chr_number[0]]
            get_ref_chrfile = get_ref_chrfile.loc[get_ref_chrfile['SNP'].isin(unique_rsids)]
            # create chr_bp column
            get_ref_chrfile['chr_bp'] = get_ref_chrfile.apply(lambda x: str(x['CHR']) + ":" + str(x['BP']), axis=1)
            get_ref_chrfile.drop_duplicates(["chr_bp"], inplace=True)

            with open(str(chrfile), 'rt') as file:
                for line_index, line in enumerate(file):

                    if line_index <= 0:
                        continue

                    arr = line.strip().split()
                    chr_bp = arr[0]
                    refallele = arr[2]
                    dosage_row = np.array(arr[3:], dtype=np.float64)

                    get_rsid = get_ref_chrfile['chr_bp'].astype(str).isin([str(chr_bp)])
                    if any(get_rsid):
                        rsid = get_ref_chrfile.loc[get_rsid]['SNP'].tolist()[0]
                        yield rsid, refallele, dosage_row
                    else:
                        continue

    @staticmethod
    def UniqueRsid(beta_file):
        res = [
            x[0] for x in WeightsDB(beta_file).query(
                "SELECT distinct rsid FROM weights")
        ]
        return res

    @staticmethod
    def get_reference(file,
                      chunkSize=1000000,
                      parition=661):
        reader = pd.read_csv(file, sep="\t", iterator=True)
        chunks = []
        with tqdm(range(parition)) as pbar:
            for _ in pbar:
                try:
                    chunk = reader.get_chunk(chunkSize)
                    chunks.append(chunk)
                except StopIteration:
                    break
        return  pd.concat(chunks, ignore_index=True)


class TranscriptionMatrix():

    def __init__(self,
                 beta_file,
                 sample_file):

        self.D = None
        self.beta_file = beta_file
        self.sample_file = sample_file
        self.complements = {"A": "T", "C": "G", "G": "C", "T": "A"}

    def update(self,
               gene,
               weight,
               ref_allele,
               allele,
               dosage_row):

        if self.D is None:

            self.gene_list = [
                tup[0] for tup in WeightsDB(self.beta_file).query(
                    "SELECT DISTINCT gene FROM weights ORDER BY gene")
            ]

            self.gene_index = {
                gene: k
                for (k, gene) in enumerate(self.gene_list)
            }

            self.D = np.zeros((len(self.gene_list), len(dosage_row)))

        if gene in self.gene_index:
            if ref_allele == allele or self.complements[ref_allele] == allele:  # assumes non-ambiguous SNPs to resolve strand issues:
                self.D[self.gene_index[gene],] += dosage_row * weight
            else:
                self.D[self.gene_index[gene],] += (2 - dosage_row) * weight  # Update all cases for that gene

    def get_samples(self):
        
        with open(self.sample_file, 'r') as samples:
            for line in samples:
                yield [line.split()[0], line.split()[1]]

    def save(self,
             pred_exp_file,
             logger):

        self.gene_list.insert(0, "FID")
        self.gene_list.insert(1, "IID")
        sample_generator = self.get_samples()


        output_dir = {
            column_name: [np.nan] * self.D.shape[1]
            for column_name in self.gene_list
        }
        output_df = pd.DataFrame(output_dir)

        for col in range(0, self.D.shape[1]):
            try:
                output_df.iloc[[col]] = next(sample_generator) + self.D[:, col].tolist()
            except Exception:
                sys.stderr.write("ERROR: There are not enough rows in your sample file!")
                sys.exit(1)
                
        try:
            next(sample_generator)
        except Exception:
            logger.info("Predicted expression file complete ...")
        else:
            sys.stderr.write("ERROR: There are too many rows in your sample file!")
            sys.exit(1)

        output_df.to_csv(pred_exp_file, index=False)
