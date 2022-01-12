.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
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

create_environment: test_environment
ifeq (True,$(HAS_CONDA)) # assume on local
	@echo ">>> Detected conda. Assume local computer. Installing packages from yml."
	bash install_conda_local.sh
else # assume on HPC
	@echo ">>> No Conda detected. Assume on HPC."
	bash hpc_install_env.sh
	@echo ">>> venv created. Activate with source ~/featstore/bin/activate"
endif


## Download data
download:
ifeq (True,$(HAS_CONDA)) # assume on local
	bash src/dataprep/download_data_local.sh $(PROJECT_DIR)
else # assume on HPC
	bash src/dataprep/download_data_hpc.sh $(PROJECT_DIR)
endif

## Make Dataset
data: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/dataprep/make_dataset.py --path_data_folder $(PROJECT_DIR)/data/
else # assume on HPC
	sbatch src/dataprep/make_raw_data_hpc.sh $(PROJECT_DIR)
endif


## Make Features
features: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/features/build_features.py --path_data_folder $(PROJECT_DIR)/data/
else # assume on HPC
	bash src/features/scripts/chain_build_feat_and_combine.sh $(PROJECT_DIR)
endif


## Select Features, Scale, and return Data Splits
splits: requirements
ifeq (True,$(HAS_CONDA)) # assume on local
	$(PYTHON_INTERPRETER) src/features/select_feat_and_scale.py --path_data_folder $(PROJECT_DIR)/data/
else # assume on HPC
	sbatch src/features/scripts/split_and_save_hpc.sh $(PROJECT_DIR)
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
