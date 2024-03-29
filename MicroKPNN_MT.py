import sys
import os
import subprocess
import argparse

######################
## PARSE ARGUMENTS ###
######################
parser = argparse.ArgumentParser(description='MicroKPNN')

parser.add_argument('--data_path', type=str, required=True, help='Path to data')
parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata')
parser.add_argument('--device', type=int, default=0, help='Which gpu to use if any (default: 0)')
parser.add_argument('--output', type=str, help='path to output folder')
parser.add_argument('--taxonomy', type=str , default=5, help='taxonomy to use for hidden layer')
parser.add_argument('--k_fold', type=int, default=5, help='k for k-fold validation')
args = parser.parse_args()

# Create sub directories in output directory
if not os.path.exists(f"{args.output}/NetworkInput"):
    cmd = f"mkdir -p {args.output}/NetworkInput"
    check_result=subprocess.check_output(cmd, shell=True)
if not os.path.exists(f"{args.output}/Record"):
    cmd = f"mkdir -p {args.output}/Record"
    check_result=subprocess.check_output(cmd, shell=True)
if not os.path.exists(f"{args.output}/Checkpoint"):
    cmd = f"mkdir -p {args.output}/Checkpoint"
    check_result=subprocess.check_output(cmd, shell=True)
print("Create/Check existance of NetworkInput dir, Record dir and Checkpoint dir in directory ", args.output )

# create specied taxonomy info
cmd = f"python lib/taxonomy_info.py --inp {args.data_path} --out {args.output}/NetworkInput/"
check_result=subprocess.check_output(cmd, shell=True)
print("create species_info.pkl")

# create edges
cmd = f"python lib/create_edges.py --inp {args.data_path} --taxonomy {args.taxonomy} --out {args.output}/NetworkInput/"
check_result=subprocess.check_output(cmd, shell=True)
print("EdgeList has been created")

# train model
edges= args.output+"/NetworkInput/EdgeList.csv"
if args.k_fold != 0: 
    # k-fold validation
    cmd = f"python lib/train_meta_kfold.py --k_fold {args.k_fold} --data_path {args.data_path} --metadata_path {args.metadata_path} --edge_list {edges} --output {args.output} --checkpoint_path {args.output}/Checkpoint/microkpnn_mt.pt --records_path {args.output}/Record/microkpnn_mt.csv --device {args.device}"
else:
    # randomly spliting the training and test sets
    cmd = f"python lib/train_meta.py --data_path {args.data_path} --metadata_path {args.metadata_path} --edge_list {edges} --output {args.output} --checkpoint_path {args.output}/Checkpoint/microkpnn_mt.pt --records_path {args.output}/Record/microkpnn_mt.csv --device {args.device}"
check_result=subprocess.check_output(cmd, shell=True)
print("Well Done")
