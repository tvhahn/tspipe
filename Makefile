.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
NOW_TIME := $(shell date +"%Y-%m-%d-%H%M-%S")
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = feat-store
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

create_environment:
ifeq (True,$(HAS_CONDA)) # assume on local
	@echo ">>> Detected conda. Assume local computer. Installing packages from yml."
	bash install_conda_local.sh
else # assume on HPC
	@echo ">>> No Conda detected. Assume on HPC."
	bash install_env_hpc.sh
	@echo ">>> venv created. Activate with source ~/featstore/bin/activate"
endif


## Download data
download:
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/dataprep/download_data.py \
	--path_data_folder $(PROJECT_DIR)/data/
else # assume on HPC
	python src/dataprep/download_data.py --path_data_folder ~/scratch/feat-store/data/
endif

## Make Dataset
data_milling: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/dataprep/make_dataset_milling.py \
	-p $(PROJECT_DIR) \
	--path_data_dir $(PROJECT_DIR)/data/ \
	--window_len 64 \
	--stride 64 \
	--raw_dir_name stride64_len64
else # assume on HPC
	sbatch src/dataprep/make_dataset_milling_hpc.sh $(PROJECT_DIR)
endif

## Make raw data for CNC
splits_cnc: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/dataprep/make_splits_cnc.py \
	-p $(PROJECT_DIR) \
	--path_data_dir $(PROJECT_DIR)/data/ \
	--save_dir_name data_splits \
	--n_cores 6
else # assume on HPC
	sbatch src/dataprep/make_splits_cnc_hpc.sh $(PROJECT_DIR)
endif


## Make raw data for CNC
data_cnc: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/dataprep/make_dataset_cnc.py \
		-p $(PROJECT_DIR) \
		--path_data_dir $(PROJECT_DIR)/data/ \
		--split_dir_name data_splits \
		--raw_dir_name data_raw_processed \
		--tool_no 54
else # assume on HPC
	sbatch src/dataprep/make_dataset_cnc_hpc.sh $(PROJECT_DIR)
endif


## Copy the raw cnc data to HPC scratch
copy_cnc_raw: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	echo "On local compute."
else # assume on HPC
	bash src/dataprep/copy_cnc_raw_to_scratch.sh
endif


## Copy the raw cnc data to HPC scratch
copy_milling_raw: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	echo "On local compute."
else # assume on HPC
	bash src/dataprep/copy_milling_raw_to_scratch.sh
endif



## Make Features
features_milling: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/features/build_features_cnc.py \
		-p $(PROJECT_DIR) \
		--dataset milling \
		--path_data_dir $(PROJECT_DIR)/data/ \
		--raw_dir_name stride64_len64 \
		--raw_file_name milling.csv.gz \
		--processed_dir_name milling_features \
		--feat_file_name milling_features.csv \
		--feat_dict_name dummy \
		--n_cores 6
else # assume on HPC
	bash src/features/chain_build_feat_and_combine_milling_hpc.sh $(PROJECT_DIR)
endif


## Make Features
features_cnc: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/features/build_features_cnc.py \
		-p $(PROJECT_DIR) \
		--dataset cnc \
		--path_data_dir $(PROJECT_DIR)/data/ \
		--raw_dir_name data_raw_processed \
		--raw_file_name cnc_raw_54.csv \
		--processed_dir_name cnc_features \
		--feat_file_name cnc_features_54.csv \
		--feat_dict_name dummy \
		--n_cores 6
else # assume on HPC
	bash src/features/chain_build_feat_and_combine_cnc_hpc.sh $(PROJECT_DIR)
endif


## Select Features, Scale, and return Data Splits
splits: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/features/select_feat_and_scale.py \
		--path_data_folder $(PROJECT_DIR)/data/
else # assume on HPC
	sbatch src/features/scripts/split_and_save_hpc.sh $(PROJECT_DIR)
endif


train_milling: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/models/train.py \
		--save_dir_name interim_results_milling \
		--processed_dir_name milling_features \
		--rand_search_iter 10 \
		--dataset milling \
		--feat_file_name milling_features.csv
else # assume on HPC
	sbatch src/models/train_milling_hpc.sh $(PROJECT_DIR)
endif


train_cnc: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/models/train.py \
		--save_dir_name interim_results_cnc \
		--processed_dir_name cnc_features_comp \
		--rand_search_iter 150 \
		--dataset cnc \
		--feat_file_name cnc_features_54_comp.csv
else # assume on HPC
	sbatch src/models/train_cnc_hpc.sh $(PROJECT_DIR)
endif



compile_milling: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/models/compile.py \
		-p $(PROJECT_DIR) \
		--n_cores 6 \
		--path_model_dir $(PROJECT_DIR)/models \
		--interim_dir_name interim_results_milling \
		--final_dir_name final_results_milling
else # assume on HPC
	sbatch src/models/compile_milling_hpc.sh $(PROJECT_DIR)
endif


compile_cnc: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/models/compile.py \
		-p $(PROJECT_DIR) \
		--n_cores 6 \
		--path_model_dir $(PROJECT_DIR)/models \
		--interim_dir_name interim_results_cnc \
		--final_dir_name final_results_cnc
else # assume on HPC
	sbatch src/models/compile_cnc_hpc.sh $(PROJECT_DIR)
endif


filter_milling: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/models/filter.py \
		-p $(PROJECT_DIR) \
		--path_data_dir $(PROJECT_DIR)/data \
		--path_model_dir $(PROJECT_DIR)/models \
		--dataset milling \
		--processed_dir_name milling_features \
		--feat_file_name milling_features.csv \
		--final_dir_name final_results_milling \
		--keep_top_n 1 \
		--save_n_figures 8
else # assume on HPC
	sbatch src/models/filter_hpc.sh $(PROJECT_DIR)
endif


filter_cnc: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/models/filter.py \
		-p $(PROJECT_DIR) \
		--path_data_dir $(PROJECT_DIR)/data \
		--path_model_dir $(PROJECT_DIR)/models \
		--dataset cnc \
		--processed_dir_name cnc_features_comp \
		--feat_file_name cnc_features_54_comp.csv \
		--final_dir_name final_results_cnc \
		--keep_top_n 1 \
		--save_n_figures 8
else # assume on HPC
	sbatch src/models/filter_hpc.sh $(PROJECT_DIR)
endif

## Make Features
viz: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/visualization/visualize.py \
		-p $(PROJECT_DIR) \
		--dataset cnc \
		--path_data_dir $(PROJECT_DIR)/data/ 
else # assume on HPC
	bash src/features/chain_build_feat_and_combine_cnc_hpc.sh $(PROJECT_DIR)
endif


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.out" -delete


## Run unit and integration tests
test:
	$(PYTHON_INTERPRETER) -m unittest discover -s tests

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
