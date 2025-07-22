import argparse
import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'     # Suppress INFO
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'    # Suppress verbose internal logs
import sys
from pathlib import Path
from cat_vs_dog.train import *
from cat_vs_dog.predict import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--list', help='List All Models', action='store_true')
	parser.add_argument('-t', '--train', help='Train Model (e.g., "1" or "catvsdog")')
	parser.add_argument('-r', '--run', help='Run Model')
	parser.add_argument('-d', '--dataset', help='Optional: path to custom dataset')
	parser.add_argument('-o', '--output', help="Path to save the trained model")
	parser.add_argument('-i', '--image', help="Path to image file")
	parser.add_argument('-m', '--model', help="Optional path to trained model (default: cat_dog_classifier.h5)")
	args = parser.parse_args()

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(0)

	if args.list:
		print('List Of Models:')
		print('---------------------')
		print('1. Cat Vs Dog (catvsdog)')
		sys.exit(0)
	
	# Use Python executable (cross-platform)
	python_exec = sys.executable
	if args.train and args.train.lower() in ('1', 'catvsdog'):
		# Path to train.py (cross-platform safe)
		train_path = Path(__file__).parent / 'cat_vs_dog' / 'train.py'
		
		if not train_path.exists():
			print(f"❌ train.py not found at '{train_path}'")
			sys.exit(1)
		
		command = [python_exec, str(train_path)]
		if args.dataset:
			command += ['--dataset', args.dataset]
		if args.output:
			command += ['--output', args.output]
		subprocess.run(command)
	
	if args.run and args.run.lower() in ('1', 'catvsdog'):
		run_path = Path(__file__).parent / 'cat_vs_dog' / 'predict.py'
		if not run_path.exists():
			print(f"❌ train.py not found at '{run_path}'")
			sys.exit(1)

		command = [python_exec, str(run_path)]
		if args.image:
			command += ['--image', args.image]
		if args.model:
			command += ['--model', args.model]
		subprocess.run(command)

if __name__ == '__main__':
	main()

"""
Folder Structure (Example if using custom dataset)

project/
├── main.py
├── cat_vs_dog/
│   └── train.py
└── my_custom_dataset/
    ├── train/
    │   ├── cat/
    │   └── dog/
    └── val/
        ├── cat/
        └── dog/


"""