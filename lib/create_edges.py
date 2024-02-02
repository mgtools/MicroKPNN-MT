import argparse
import csv
import networkx as nx
import pandas as pd
import pickle
import os


def load_taxonomy_info(file_path):
    with open(file_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

def read_gml(file_path):
    return nx.read_gml(file_path)

def read_metabolite_file(file_path):
    metabolite = []
    with open(file_path, newline='') as metabolite_file:
        line_reader = csv.reader(metabolite_file, delimiter='\t')
        for line in line_reader:
            metabolite.append(line)
    return metabolite

def clean_node_name(name):
    cleaned_name = ''.join(name.split(' '))
    cleaned_name = ''.join(cleaned_name.split(','))
    cleaned_name = ''.join(cleaned_name.split('('))
    cleaned_name = ''.join(cleaned_name.split(')'))
    cleaned_name = ''.join(cleaned_name.split('['))
    cleaned_name = ''.join(cleaned_name.split(']'))
    cleaned_name = ''.join(cleaned_name.split('&'))
    cleaned_name = ''.join(cleaned_name.split('+'))
    return cleaned_name

def create_edges_from_taxonomy(species_ids, taxonomy_infos, taxonomy_num):
    edges = []
    taxonomy_nodes = []
    count = 0
    taxa = {0:'superkingdom', 1:'phylum',2:'class', 3:'order', 4:'family', 5:'genus'}
    taxonomy_field = taxa[int(taxonomy_num)]
    for species_id in species_ids:
        if taxonomy_infos[species_id][taxonomy_field] != '':
            edges.append((taxonomy_infos[species_id][taxonomy_field], species_id))
            taxonomy_nodes.append(taxonomy_infos[species_id][taxonomy_field])
        
    print(list(set(taxonomy_nodes)))
    print(f"Number of {taxonomy_field}s: {len(set(taxonomy_nodes))}")
    return edges, taxonomy_nodes

def create_edges_from_graph(species_ids, G):
    edges = []
    gspecies_id = []
    gpartition = []

    for n in G.nodes:
        gspecies_id.append(G.nodes[n]["name"])
        gpartition.append(G.nodes[n]["leidenpartition"])

    community_nodes = []
    for species_id in species_ids:
        for i in range(len(gspecies_id)):
            if species_id == gspecies_id[i]:
                edges.append((gpartition[i], species_id))
                community_nodes.append(gpartition[i])
                break

    print(f"Number of communities: {len(set(community_nodes))}")
    return edges, community_nodes

def create_edges_from_metabolites(species_ids, taxonomy_infos, metabolite):
    edges = []
    metabolite_nodes = []

    for species_id in species_ids:
        for m in metabolite:
            if m[2] == taxonomy_infos[species_id]['species']:
                parent = clean_node_name(m[0] + ' ' + m[1])
                edges.append((parent, species_id))
                metabolite_nodes.append(parent)

    print(f"Number of metabolites: {len(set(metabolite_nodes))}")
    return edges, metabolite_nodes

def create_edges_for_hidden(species_ids, num_hidden):
    edges = []
    meaningless_nodes = []

    for i in range(num_hidden):
        for species_id in species_ids:
            parent = 'h' + str(i)
            meaningless_nodes.append(parent)
            edges.append((parent, species_id))

    print(f"Number of hidden nodes: {len(set(meaningless_nodes))}")
    return edges, meaningless_nodes

def create_output_edges(edges):
    output_edges = []

    for e in edges:
        output_edges.extend([
            ('phenotype', e[0]),
            ('age', e[0]),
            ('gender', e[0]),
            ('BMI', e[0])
        ])

    output_edges = list(set(output_edges))
    return output_edges

def save_nodes_to_csv(nodes, file_path):
    df = pd.DataFrame({'nodes': nodes})
    df.to_csv(file_path, index=False)

def save_edges_to_csv(edges, file_path):
    parents, children = zip(*edges)
    df = pd.DataFrame({'parent': parents, 'child': children})
    df.to_csv(file_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, required=True)
    parser.add_argument('--taxonomy', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    Default_Database = "Default_Database/"
    community_file = Default_Database + "WT_spiec-easi.filtered.annotated.gml"
    metabolite_file = Default_Database + "NJS16_metabolite_species_association.txt"

    df_relative_abundance = pd.read_csv(args.inp, index_col=0)
    species_ids = df_relative_abundance.columns.values.tolist()


    file_in_parent_directory = os.path.join(args.out + '/species_info.pkl')
    print(file_in_parent_directory)
    taxonomy_infos = load_taxonomy_info(file_in_parent_directory)

    # Create edges
    taxonomy_edges, taxonomy_nodes = create_edges_from_taxonomy(species_ids, taxonomy_infos, args.taxonomy)
    community_edges, community_nodes = create_edges_from_graph(species_ids, read_gml(community_file))
    metabolite_edges, metabolite_nodes = create_edges_from_metabolites(species_ids, taxonomy_infos, read_metabolite_file(metabolite_file))

    output_edges = create_output_edges(taxonomy_edges + community_edges + metabolite_edges)
    all_edges = output_edges + taxonomy_edges + community_edges + metabolite_edges
    
    # Save nodes to CSV
    save_nodes_to_csv(list(set(taxonomy_nodes)), args.out + '/taxonomyNodes.csv')
    save_nodes_to_csv(list(set(community_nodes)), args.out + '/communityNodes.csv')
    save_nodes_to_csv(list(set(metabolite_nodes)), args.out + '/metaboliteNodes.csv')

    # Save edges to CSV
    save_edges_to_csv(all_edges, args.out+'/EdgeList.csv')

if __name__ == "__main__":
    main()
