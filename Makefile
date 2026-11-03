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
STEPS ?= 500000
SNAPSHOT_INTERVAL ?= 100000
JOBS ?= 4
ARGS ?=
CONFIRM ?=

# Runs of a different length get their own folder, so they never mix with the 500,000-step study.
ifeq ($(STEPS),500000)
LOG_DIR ?= logs
else
LOG_DIR ?= logs/steps_$(STEPS)
endif

ifeq ($(LOG_DIR),logs)
FIGURES_DIR := docs/figures
CLEAN_PATHS := $(addprefix $(LOG_DIR)/,csv tensorboard checkpoints wandb videos run_logs study.log)
KEPT_MESSAGE := Kept: logs/pilot, longer runs in logs/steps_*, and docs/figures.
else
FIGURES_DIR := $(LOG_DIR)/figures
CLEAN_PATHS := $(addprefix $(LOG_DIR)/,csv tensorboard checkpoints wandb videos run_logs study.log figures)
KEPT_MESSAGE := Kept: everything outside $(LOG_DIR).
endif

PY := uv run python -m
PILOT_DIR := logs/pilot
TRIAL_DIR := logs/smoke

.PHONY: help usage setup wandb-login test lint format typecheck check trial trial-clean baseline \
	pilot pilot-clean train study status tensorboard sync figures videos clean-run clean-results

##@ Help

help: ## Show this list of commands
	@if [ -t 1 ] && [ -z "$${NO_COLOR:-}" ]; then \
		bold=$$'\033[1m'; dim=$$'\033[2m'; cyan=$$'\033[36m'; \
		yellow=$$'\033[33m'; green=$$'\033[32m'; reset=$$'\033[0m'; \
	else \
		bold=""; dim=""; cyan=""; yellow=""; green=""; reset=""; \
	fi; \
	var() { printf "  %s%-18s%s %s %s[%s]%s\n" "$$yellow" "$$1" "$$reset" "$$2" "$$dim" "$$3" "$$reset"; }; \
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
	var STEPS "Training steps per run for train and study" '$(STEPS)'; \
	var JOBS "Runs to train at the same time in study" '$(JOBS)'; \
	var SNAPSHOT_INTERVAL "Steps between model snapshots" '$(SNAPSHOT_INTERVAL)'; \
	var LOG_DIR "Run outputs; logs/steps_<STEPS> if STEPS changes" '$(LOG_DIR)'; \
	var ARGS "Extra options passed to the underlying command" '$(ARGS)'; \
	printf "  %s%-18s%s %s\n" "$$yellow" RUN "$$reset" "Run to delete with clean-run, such as ddqn_n3_seed1"; \
	printf "  %s%-18s%s %s\n" "$$yellow" CONFIRM "$$reset" "Set to yes to skip the clean-results question"; \
	printf "\n%sExamples:%s\n" "$$bold" "$$reset"; \
	example 'make study' "Train all 20 runs"; \
	example 'make train VARIANT=ddqn_n5 SEED=2' "Train one run"; \
	example 'make study JOBS=4' "Train 4 runs at a time"; \
	example 'make study STEPS=2000000' "Train all 20 runs for 2 million steps"; \
	example 'make videos ARGS="--fps 15"' "Record videos at real game speed"; \
	example 'make test ARGS="-k polyak"' "Run tests matching a word"; \
	example 'make clean-results' "Delete all study results, after asking"; \
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
		--wandb-mode $(WANDB) --log-dir "$(LOG_DIR)" --total-timesteps $(STEPS) \
		--snapshot-interval $(SNAPSHOT_INTERVAL) $(ARGS)

study: ## Train all 20 runs, skipping finished ones; JOBS=4 trains 4 at a time
	$(PY) scripts.study --jobs $(JOBS) --variants $(VARIANTS) --seeds $(SEEDS) \
		--device $(DEVICE) --wandb-mode $(WANDB) --log-dir "$(LOG_DIR)" \
		--total-timesteps $(STEPS) --snapshot-interval $(SNAPSHOT_INTERVAL) $(ARGS)

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

figures: ## Make the figures from LOG_DIR's runs (docs/figures for the study)
	$(PY) scripts.plot_results --csv-dir "$(LOG_DIR)/csv" \
		--output "$(FIGURES_DIR)/ablation_comparison.png" \
		--overestimation-output "$(FIGURES_DIR)/overestimation_evaluation.png" $(ARGS)

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

clean-results: ## Delete every result in LOG_DIR to restart training, after asking first
	@case "$(LOG_DIR)" in \
		"" | . | /* | *..*) echo "Refusing to clean LOG_DIR=$(LOG_DIR): use a folder inside the project."; exit 2 ;; \
	esac; \
	if pgrep -f -- "--log-dir $(LOG_DIR)( |$$)" > /dev/null; then \
		echo "Training is still writing to $(LOG_DIR). Stop it before deleting its results."; \
		exit 1; \
	fi; \
	targets=""; \
	for path in $(CLEAN_PATHS); do \
		if [ -e "$$path" ]; then targets="$$targets $$path"; fi; \
	done; \
	if [ -z "$$targets" ]; then echo "No results to delete in $(LOG_DIR)."; exit 0; fi; \
	echo "This permanently deletes:"; \
	du -sh $$targets | sed 's/^/  /'; \
	echo "$(KEPT_MESSAGE)"; \
	if [ "$(CONFIRM)" != "yes" ]; then \
		if [ ! -t 0 ]; then \
			echo "Can't ask for confirmation here. Run with CONFIRM=yes to delete without asking."; \
			exit 1; \
		fi; \
		printf "Type yes to delete them: "; \
		read -r answer; \
		if [ "$$answer" != "yes" ]; then echo "Nothing was deleted."; exit 0; fi; \
	fi; \
	rm -rf $$targets; \
	echo "Deleted. Run make study to train again."
