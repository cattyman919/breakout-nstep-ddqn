# Commands for the whole project. Run `make help` to list them.
#
# Lines marked `## text` are read by the help target, so a new command documents itself.
# This file avoids features newer than GNU Make 3.81, the version macOS ships.

SHELL := /bin/bash
.DEFAULT_GOAL := help

VARIANT ?= ddqn_n3
SEED ?= 1
VARIANTS ?= dqn_n1 ddqn_n1 ddqn_n3 ddqn_n5
SEEDS ?= 1 2 3 4 5
DEVICE ?= cpu
WANDB ?= offline
LOG_DIR ?= logs
ARGS ?=

PY := uv run python -m
PILOT_DIR := logs/pilot
TRIAL_DIR := logs/smoke

.PHONY: help usage setup wandb-login test lint format typecheck check trial trial-clean baseline \
	pilot pilot-clean train study status tensorboard sync figures videos clean-run

##@ Help

help: ## Show this list of commands
	@if [ -t 1 ] && [ -z "$${NO_COLOR:-}" ]; then \
		bold=$$'\033[1m'; dim=$$'\033[2m'; cyan=$$'\033[36m'; \
		yellow=$$'\033[33m'; green=$$'\033[32m'; reset=$$'\033[0m'; \
	else \
		bold=""; dim=""; cyan=""; yellow=""; green=""; reset=""; \
	fi; \
	var() { printf "  %s%-14s%s %s %s[%s]%s\n" "$$yellow" "$$1" "$$reset" "$$2" "$$dim" "$$3" "$$reset"; }; \
	example() { printf "  %s%-41s%s %s\n" "$$green" "$$1" "$$reset" "$$2"; }; \
	printf "%sUsage:%s make %s<command>%s %s[VARIABLE=value ...]%s\n" \
		"$$bold" "$$reset" "$$cyan" "$$reset" "$$yellow" "$$reset"; \
	awk -v bold="$$bold" -v cyan="$$cyan" -v reset="$$reset" 'BEGIN { FS = ":.*## " } \
		/^##@ / { printf "\n%s%s:%s\n", bold, substr($$0, 5), reset } \
		/^[a-z][a-z-]*:.*## / { printf "  %s%-14s%s %s\n", cyan, $$1, reset, $$2 }' $(MAKEFILE_LIST); \
	printf "\n%sVariables (current value in brackets):%s\n" "$$bold" "$$reset"; \
	var VARIANT "Variant for train" '$(VARIANT)'; \
	var SEED "Seed for train and videos" '$(SEED)'; \
	var VARIANTS "Variants for study, status and videos" '$(VARIANTS)'; \
	var SEEDS "Seeds for study and status" '$(SEEDS)'; \
	var DEVICE "cpu or mps; use cpu for the study" '$(DEVICE)'; \
	var WANDB "W&B mode, offline or online" '$(WANDB)'; \
	var LOG_DIR "Root folder for run outputs" '$(LOG_DIR)'; \
	var ARGS "Extra options passed to the underlying command" '$(ARGS)'; \
	printf "  %s%-14s%s %s\n" "$$yellow" RUN "$$reset" "Run to delete with clean-run, such as ddqn_n3_seed1"; \
	printf "\n%sExamples:%s\n" "$$bold" "$$reset"; \
	example 'make study' "Train all 20 runs"; \
	example 'make train VARIANT=ddqn_n5 SEED=2' "Train one run"; \
	example 'make videos ARGS="--fps 15"' "Record videos at real game speed"; \
	example 'make test ARGS="-k polyak"' "Run tests matching a word"; \
	example 'make train ARGS=--help' "List every training option"

usage: help ## Same as help

##@ Setup

setup: ## Install Python and all dependencies
	uv sync

wandb-login: ## Log in to Weights & Biases, needed only for WANDB=online
	uv run wandb login

##@ Tests and code checks

test: ## Run the tests (about 10 seconds)
	$(PY) pytest $(ARGS)

lint: ## Check the code with Ruff, without changing it
	uv run ruff check .
	uv run ruff format --check .

format: ## Fix lint problems and format the code with Ruff
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Check types with mypy
	uv run --with mypy mypy --strict --ignore-missing-imports src scripts

check: lint typecheck test ## Run every check; do this before committing

##@ Trial and pilot

trial: ## Try the whole workflow on a short run in logs/smoke (under a minute)
	rm -rf $(TRIAL_DIR)
	$(PY) scripts.train --variant ddqn_n3 --seed 1 --device $(DEVICE) --wandb-mode offline \
		--log-dir $(TRIAL_DIR) --total-timesteps 30000 --learning-starts 1000 \
		--eval-interval 10000 --eval-games 1 --snapshot-interval 10000
	$(PY) scripts.plot_results --csv-dir $(TRIAL_DIR)/csv --output $(TRIAL_DIR)/comparison.png \
		--overestimation-output $(TRIAL_DIR)/overestimation.png
	$(PY) scripts.record_videos --variants ddqn_n3 --log-dir $(TRIAL_DIR)

trial-clean: ## Delete the trial run's files
	rm -rf $(TRIAL_DIR)

baseline: ## Measure the random-policy return
	$(PY) scripts.measure_random $(ARGS)

pilot: baseline ## Run the two seed-0 pilots in logs/pilot (about 6 minutes)
	$(PY) scripts.train --variant ddqn_n3 --seed 0 --device $(DEVICE) --wandb-mode offline \
		--log-dir $(PILOT_DIR)
	$(PY) scripts.train --variant dqn_n1 --seed 0 --device $(DEVICE) --wandb-mode offline \
		--log-dir $(PILOT_DIR)

pilot-clean: ## Delete the pilot's results so it can be repeated
	rm -rf $(PILOT_DIR)

##@ Training

train: ## Train one run, chosen with VARIANT and SEED
	$(PY) scripts.train --variant $(VARIANT) --seed $(SEED) --device $(DEVICE) \
		--wandb-mode $(WANDB) --log-dir $(LOG_DIR) $(ARGS)

study: ## Train all 20 runs, skipping finished ones (a little over an hour)
	@mkdir -p "$(LOG_DIR)"; \
	failed=""; \
	for variant in $(VARIANTS); do \
		for seed in $(SEEDS); do \
			run="$${variant}_seed$${seed}"; \
			if [ -f "$(LOG_DIR)/checkpoints/$$run.pt" ]; then \
				echo "Skipping $$run: already finished"; \
				continue; \
			fi; \
			echo "=== Training $$run"; \
			$(PY) scripts.train --variant "$$variant" --seed "$$seed" --device $(DEVICE) \
				--wandb-mode $(WANDB) --log-dir "$(LOG_DIR)" $(ARGS) 2>&1 \
				| tee -a "$(LOG_DIR)/study.log"; \
			if [ "$${PIPESTATUS[0]}" -ne 0 ]; then failed="$$failed $$run"; fi; \
		done; \
	done; \
	if [ -n "$$failed" ]; then \
		echo "These runs did not finish:$$failed"; \
		echo "See $(LOG_DIR)/study.log and the Troubleshooting section of README.md."; \
		exit 1; \
	fi; \
	echo "All runs finished."

status: ## Show which study runs are finished, incomplete or not started
	@finished=0; total=0; \
	for variant in $(VARIANTS); do \
		for seed in $(SEEDS); do \
			run="$${variant}_seed$${seed}"; \
			total=$$((total + 1)); \
			if [ -f "$(LOG_DIR)/checkpoints/$$run.pt" ]; then \
				state="finished"; finished=$$((finished + 1)); \
			elif [ -f "$(LOG_DIR)/csv/$${run}_metrics.csv" ]; then \
				state="incomplete"; \
			else \
				state="not started"; \
			fi; \
			printf "  %-18s %s\n" "$$run" "$$state"; \
		done; \
	done; \
	echo "$$finished of $$total runs finished."

tensorboard: ## Open training charts at http://localhost:6006
	uv run tensorboard --logdir "$(LOG_DIR)/tensorboard"

sync: ## Upload offline runs to Weights & Biases
	uv run wandb sync $(LOG_DIR)/wandb/offline-run-*

##@ Results

figures: ## Make the figures in docs/figures from logs/csv
	$(PY) scripts.plot_results $(ARGS)

videos: ## Record videos of each variant's saved models, for SEED
	$(PY) scripts.record_videos --variants $(VARIANTS) --seed $(SEED) --log-dir "$(LOG_DIR)" $(ARGS)

##@ Cleanup

clean-run: ## Delete one run's results so it can be redone; needs RUN=name
	@case "$(RUN)" in \
		*_seed[0-9]*) ;; \
		*) echo "Choose a run to delete, such as: make clean-run RUN=ddqn_n3_seed1"; exit 2 ;; \
	esac; \
	case "$(RUN)" in \
		*[!a-z0-9_]*) echo "Not a valid run name: $(RUN)"; exit 2 ;; \
	esac
	rm -rf "$(LOG_DIR)/csv/$(RUN)_metrics.csv" "$(LOG_DIR)/tensorboard/$(RUN)" \
		"$(LOG_DIR)/checkpoints/$(RUN).pt" "$(LOG_DIR)/checkpoints/$(RUN)"
