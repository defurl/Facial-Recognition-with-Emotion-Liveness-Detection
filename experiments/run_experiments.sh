#!/bin/bash
# ==============================================================================
# Background Training Launcher for TDA Experiments
# ==============================================================================
# This script launches experiments in the background using nohup so they
# continue running after SSH disconnection.
#
# Usage:
#   ./experiments/run_experiments.sh [exp1] [exp2] ...
#   ./experiments/run_experiments.sh all          # Run all experiments
#   ./experiments/run_experiments.sh a1           # Run only exp_a1
#   ./experiments/run_experiments.sh a1 a2        # Run a1 and a2
#
# Monitor progress:
#   tail -f experiments/exp_a1_lambda_005/outputs/train.log
#   tail -f experiments/exp_a2_lambda_002/outputs/train.log
#
# Check running processes:
#   ps aux | grep -E "exp_a[12]" | grep -v grep
#
# Stop experiments:
#   pkill -f "exp_a1_lambda_005/train.py"
#   pkill -f "exp_a2_lambda_002/train.py"
# ==============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Conda environment
CONDA_ENV="face_recog"

# Experiment directories
EXP_A1_DIR="$SCRIPT_DIR/exp_a1_lambda_005"
EXP_A2_DIR="$SCRIPT_DIR/exp_a2_lambda_002"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}       TDA EXPERIMENTS - BACKGROUND TRAINING LAUNCHER                ${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "${YELLOW}Project Root:${NC} $PROJECT_ROOT"
echo -e "${YELLOW}Conda Env:${NC} $CONDA_ENV"
echo ""

# Function to check if experiment is already running
check_running() {
    local exp_name=$1
    if pgrep -f "$exp_name/train.py" > /dev/null 2>&1; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

# Function to launch an experiment
launch_experiment() {
    local exp_dir=$1
    local exp_name=$(basename "$exp_dir")
    local log_file="$exp_dir/outputs/train.log"
    local nohup_file="$exp_dir/outputs/nohup.out"
    
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Experiment: ${NC}$exp_name"
    
    # Check if already running
    if check_running "$exp_name"; then
        echo -e "${RED}⚠ Already running! Skipping...${NC}"
        echo -e "  Monitor: tail -f $log_file"
        return
    fi
    
    # Check if train.py exists
    if [ ! -f "$exp_dir/train.py" ]; then
        echo -e "${RED}✗ train.py not found in $exp_dir${NC}"
        return
    fi
    
    # Clear previous log (optional - comment out to append)
    # > "$log_file"
    
    echo -e "${GREEN}Starting training in background...${NC}"
    
    # Launch with nohup
    cd "$PROJECT_ROOT"
    nohup conda run -n "$CONDA_ENV" --no-capture-output \
        python "$exp_dir/train.py" \
        > "$nohup_file" 2>&1 &
    
    local pid=$!
    echo -e "  PID: $pid"
    echo -e "  Log: $log_file"
    echo -e "  Nohup: $nohup_file"
    echo -e "${GREEN}✓ Launched successfully${NC}"
    
    # Save PID to file for easy tracking
    echo "$pid" > "$exp_dir/outputs/train.pid"
}

# Function to show status
show_status() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}EXPERIMENT STATUS:${NC}"
    
    for exp_dir in "$EXP_A1_DIR" "$EXP_A2_DIR"; do
        local exp_name=$(basename "$exp_dir")
        if check_running "$exp_name"; then
            local pid=$(pgrep -f "$exp_name/train.py" | head -1)
            echo -e "  ${GREEN}● $exp_name${NC} - Running (PID: $pid)"
        else
            echo -e "  ${RED}○ $exp_name${NC} - Not running"
        fi
    done
}

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [exp1] [exp2] ... | all | status"
    echo ""
    echo "Available experiments:"
    echo "  a1  - exp_a1_lambda_005 (λ=0.05, 30 epochs)"
    echo "  a2  - exp_a2_lambda_002 (λ=0.02, 50 epochs)"
    echo "  all - Run all experiments"
    echo "  status - Show running experiments"
    echo ""
    show_status
    exit 0
fi

# Handle status command
if [ "$1" == "status" ]; then
    show_status
    exit 0
fi

# Determine which experiments to run
RUN_A1=false
RUN_A2=false

for arg in "$@"; do
    case $arg in
        all)
            RUN_A1=true
            RUN_A2=true
            ;;
        a1|A1|exp_a1|exp_a1_lambda_005)
            RUN_A1=true
            ;;
        a2|A2|exp_a2|exp_a2_lambda_002)
            RUN_A2=true
            ;;
        *)
            echo -e "${RED}Unknown experiment: $arg${NC}"
            ;;
    esac
done

# Launch selected experiments
if [ "$RUN_A1" = true ]; then
    launch_experiment "$EXP_A1_DIR"
    sleep 2  # Small delay between launches
fi

if [ "$RUN_A2" = true ]; then
    launch_experiment "$EXP_A2_DIR"
fi

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Experiments launched in background!${NC}"
echo ""
echo -e "${YELLOW}Monitor progress:${NC}"
echo "  tail -f $EXP_A1_DIR/outputs/train.log"
echo "  tail -f $EXP_A2_DIR/outputs/train.log"
echo ""
echo -e "${YELLOW}Check GPU usage:${NC}"
echo "  watch -n 1 nvidia-smi"
echo ""
echo -e "${YELLOW}Check status:${NC}"
echo "  $0 status"
echo ""
echo -e "${YELLOW}Stop experiments:${NC}"
echo "  pkill -f 'exp_a1_lambda_005/train.py'"
echo "  pkill -f 'exp_a2_lambda_002/train.py'"
echo ""
echo -e "${BLUE}You can now safely disconnect from SSH!${NC}"
echo -e "${BLUE}======================================================================${NC}"
