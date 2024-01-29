import pandas as pd
import pickle
import argparse
import os

def parse_nodes(nodes_file):
    nodes = {}
    with open(nodes_file, 'r') as file:
        for line in file:
            data = line.strip().split('|')
            tax_id, parent_tax_id, rank = map(str.strip, data[:3])
            nodes[tax_id] = {'parent': parent_tax_id, 'rank': rank}
    return nodes

def parse_names(names_file):
    names = {}
    with open(names_file, 'r') as file:
        for line in file:
            data = line.strip().split('|')
            tax_id, name, _, name_type = map(str.strip, data[:4])
            if name_type == 'scientific name':
                names[tax_id] = name
    return names

def get_taxonomy_info(ncbi_id, nodes, names):
    taxonomy_info = {'superkingdom': '', 'phylum': '', 'class': '', 'order': '', 'family': '', 'genus': '', 'species': ''}

    current_tax_id = ncbi_id
    while current_tax_id in nodes:
        rank = nodes[current_tax_id]['rank']
        if rank in taxonomy_info:
            taxonomy_info[rank] = names.get(current_tax_id, '')
        current_tax_id = nodes[current_tax_id]['parent']
        if current_tax_id == '1':
            break

    return taxonomy_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    Default_Database = "Default_Database/"
    nodes_file = Default_Database + "nodes.dmp"
    names_file = Default_Database + "names.dmp"
    
    df_relative_abundance = pd.read_csv(args.inp, index_col=0)
    species_ids = df_relative_abundance.columns.values.tolist()
    print('hi')
    print(os.getcwd())
    nodes = parse_nodes(nodes_file)
    names = parse_names(names_file)

    taxonomy_infos = {}
    for ncbi_id in species_ids:
        taxonomy_info = get_taxonomy_info(ncbi_id, nodes, names)
        taxonomy_infos[ncbi_id] = taxonomy_info

    with open(args.out + '/species_info.pkl', 'wb') as pickle_file:
        pickle.dump(taxonomy_infos, pickle_file)

if __name__ == "__main__":
    main()
