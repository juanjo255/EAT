# Changes in this fork

- **FASTA queries vs. pre-computed H5 lookup (`eat.py`)**: embeddings computed on the fly from a FASTA file come out of ProtT5 in half precision (float16), while embeddings read from an H5 file are cast to float32. Mixing both (e.g. `-l sprot.h5 -q queries.fasta`) crashed `torch.cdist` with `RuntimeError: expected scalar type Half but found Float`. FASTA-derived embeddings are now cast to float32 when returned (`read_inputs()`); the H5 file written to the output folder is unchanged.
- **`annotate_eat_results.py`**: fills the `Lookup-Label` column of `eat_result.txt` with UniProt annotations (see [Annotating hits from the SwissProt lookup](#annotating-hits-from-the-swissprot-lookup)).
- **Updated SwissProt lookup**: the 2021 SwissProt embeddings were replaced by the current UniProt release (see [Updated SwissProt embeddings from UniProt](#updated-swissprot-embeddings-from-uniprot)).
- **`.gitignore`**: local backups (`*.bak*`), the HuggingFace cache (`hf_cache/`), and large embedding/weight files (`data/lookup_dbs/`, `*.h5`, `tucker_weights.pt`) are not tracked.

# EAT

Embedding-based annotation transfer (EAT) uses Euclidean distance between vector representations (embeddings) of proteins to transfer annotations from a set of labeled lookup protein embeddings to query protein embeddings.

# Abstract

Here, we present a novel approach that expands the concept of homology-based inference (HBI) from a low-dimensional sequence-distance lookup to the level of a high-dimensional embedding-based annotation transfer (EAT). More specifically, we replace sequence similarity as means to transfer annotations from one set of proteins (lookup; usually labeled) to another set of proteins (queries; usually unlabeled) by Euclidean distance between single protein sequence representations (embeddings) from protein Language Models (pLMs). Secondly, we introduce a novel set of embeddings (dubbed ProtTucker) that were optimized towards constraints captured by hierarchical classifications of protein 3D structures (CATH). These new embeddings enabled the intrusion into the midnight zone of protein comparisons, i.e., the region in which the level of pairwise sequence similarity is akin of random relations and therefore is hard to navigate by HBI methods. Cautious benchmarking showed that ProtTucker reached further than advanced sequence comparisons without the need to compute alignments allowing it to be orders of magnitude faster.

# Getting started

Clone the EAT repository and get started as described in the Usage section below:

```sh
git clone https://github.com/Rostlab/EAT.git
```

Set up a local python virtual environment and install dependencies:

```sh
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

# Usage

- Quick start: General purpose (1 nearest neighbor/1-NN) without additional labels files:

For general annotation transfer/nearest-neighbor search in embedding space, the pLM ProtT5 is used. It was only optimized using raw protein sequences (self-supervised pre-training) and is therefor not biased towards a certain task. The following command will take two FASTA files holding protein sequences as input (lookup & queries) in order to transfer annotations (fasta headers) from lookup to queries:

```sh
python eat.py --lookup data/example_data_subcell/deeploc_lookup.fasta --queries data/example_data_subcell/la_query_setHARD.fasta --output eat_results/
```

- Extended: General purpose (3-NN) with additional labels:

If you want to provide your labels as separate file (labels are expected to have CSV format with 1st col being the fasta header and 2nd col being the label) and retrieve the first 3 nearest-neighbors (NN) instead of only the single NN:

```sh
python eat.py --lookup data/example_data_subcell/deeploc_lookup.fasta --queries data/example_data_subcell/la_query_setHARD.fasta --output eat_results/ --lookupLabels data/example_data_subcell/deeploc_lookup_labels.txt --queryLabels data/example_data_subcell/la_query_setHARD_labels.txt --num_NN 3
```

Example output is given here: [Example output](https://github.com/Rostlab/EAT/blob/main/data/example_data_subcell/example_output_protT5_NN3.txt)

- Expert solution tailored for remote homology detection:

For remote homology detection, we recommend to use ProtTucker(ProtT5) embeddings that were specialized on capturing the CATH hierarchy. Additionally, we set a maximum Euclidean distance threshold of 1.1, i.e., hits with Euclidean distance larger than 1.1 are not transferred :

```sh
python eat.py --lookup data/example_data_subcell/deeploc_lookup.fasta --queries data/example_data_subcell/la_query_setHARD.fasta --output eat_results/ --use_tucker 1 --threshold 1.1
```

# Pre-computed lookup embeddings

We have pre-computed embeddings for major databases to simplify your EAT search (release date of DBs: 16.11.2021):

- SwissProt (565k proteins, 1.3GB): [Download SwissProt embeddings](https://rostlab.org/~deepppi/eat_dbs/sprot_161121.h5)
- PDB (668k chains, 1.6GB): [Download PDB embeddings](https://rostlab.org/~deepppi/eat_dbs/pdb_seqres_161121.h5)
- CATH-S100 v4.3 (122k chains, 286MB): [Download CATH embeddings](https://rostlab.org/~deepppi/eat_dbs/cath_v430_dom_seqs_S100_161121.h5)
- SCOPe v.2.08 (93k chains, 216MB): [Download SCOPe embeddings](https://rostlab.org/~deepppi/eat_dbs/scope_2.08_S100.h5)

All embeddings listed above were generated using ProtT5-XL-U50 (or in short ProtT5).
In a first step, per-residue embeddings (Lx1024 for ProtT5) were computed.
Per-protein embeddings were derived by averaging over the per-residue embeddings, resulting in a single 1024-d vector for each protein, irrespective of its length.
Embeddings are stored as H5 files with protein/chain identifiers (either SwissProt-, PDB-, CATH-, or SCOPe-IDs) as keys and 1024-d embeddings as values.

The model was run in half-precision mode on a Quadro RTX 8000 with 48GB vRAM.
Proteins longer than 9.1k residues had to be excluded due to OOM-errors (only a handful proteins were affected by this).

The embeddings can readily be used as input to the lookup file parameter of EAT.
IF you want to detect structural homologs, we recommend running the provided lookup embeddings through ProtTucker(ProtT5) first. This can be done by adding the flag `--use_tucker 1`.

## Updated SwissProt embeddings from UniProt

UniProt distributes per-protein ProtT5 embeddings for UniProtKB/Swiss-Prot with every release (https://www.uniprot.org/help/downloads#embeddings).
They are generated with the same model (ProtT5-XL-U50, mean-pooled 1024-d, half precision) and keyed by UniProt accession, so they are a drop-in replacement for `sprot_161121.h5`.
Proteins longer than 12k residues are excluded by UniProt.

```sh
mkdir -p data/lookup_dbs
wget -c https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/uniprot_sprot/per-protein.h5 \
     -O data/lookup_dbs/sprot_2026_09.h5
python eat.py -l data/lookup_dbs/sprot_2026_09.h5 -q queries.fasta -o eat_results/ --num_NN 2
```

Version used here: UniProt release of 2026-09-03 (`sprot_2026_09.h5`, 575,736 proteins, 1.3 GB), compared with 565,232 proteins in `sprot_161121.h5`.
Nearest neighbors can differ from the 2021 DB because of the newer entries, but distances are on the same scale, so thresholds (e.g. `--threshold 1.1`) keep their meaning.
`data/lookup_dbs/` is git-ignored; download the file after cloning.

## Annotating hits from the SwissProt lookup

The SwissProt embeddings only store accessions, so without `--lookupLabels` the `Lookup-Label` column of `eat_result.txt` is `None`.
`annotate_eat_results.py` fills it post hoc from the UniProt REST API (protein name by default) and appends organism and gene names, without re-running EAT:

```sh
python annotate_eat_results.py -i eat_results/eat_result.txt
# -> eat_results/eat_result_annotated.tsv
python annotate_eat_results.py -i eat_results/eat_result.txt -o annotated.tsv --fields protein_name,organism_name,gene_names,ec
```

Accessions that became secondary since the lookup DB was built (relevant for the 2021 snapshot) are resolved to their current entries (listed in `Current-Accession`); IDs not found in UniProt are written as `NA`.

# Train your own network

The following steps will allow you to replicate the training of ProtTucker:
First, clone this repository and install dependencies:

```sh
git clone https://github.com/Rostlab/EAT.git
```

Next, download pre-computed embeddings used in the paper (ProtT5, ProtBERT, ESM-1b, ProSE: 5.5GB in total) to data/ProtTucker and unzip them. Also, download CATH annotations used for training.

```sh
wget -P data/ProtTucker/ https://rostlab.org/~deepppi/prottucker_training_embeddings.tar.gz
tar -xvf data/ProtTucker/prottucker_training_embeddings.tar.gz -C data/ProtTucker/ --strip-components 1
wget -P data/ProtTucker https://rostlab.org/~deepppi/cath-domain-list.txt
```

Finally, start training by running the training script:

```sh
python train_prottucker.py
```

By default, this will train ProtTucker as reported in the paper using embeddings from ProtT5.
In order to change the input embeddings, you can either replace the file name for 'embedding_p' OR compute your own embeddings (supported input format: H5).

# Reference

```Bibtex
@article{10.1093/nargab/lqac043,
    author = {Heinzinger, Michael and Littmann, Maria and Sillitoe, Ian and Bordin, Nicola and Orengo, Christine and Rost, Burkhard},
    title = "{Contrastive learning on protein embeddings enlightens midnight zone}",
    journal = {NAR Genomics and Bioinformatics},
    volume = {4},
    number = {2},
    year = {2022},
    month = {06},
    issn = {2631-9268},
    doi = {10.1093/nargab/lqac043},
    url = {https://doi.org/10.1093/nargab/lqac043},
    note = {lqac043},
    eprint = {https://academic.oup.com/nargab/article-pdf/4/2/lqac043/44245898/lqac043.pdf},
}
```
