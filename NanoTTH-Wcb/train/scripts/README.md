# Dataset Sampler & Merger

This Python utility automates the process of randomly sampling ROOT files from a large dataset, moving them to a training directory, and optionally merging them into single files per category.

It is designed to maintain a history of moved files to ensure reproducibility and allow for an easy "reset" (moving files back to their original location).

## Features

1.  **Config-Driven**: Defines dataset categories and file prefixes via a YAML configuration.
2.  **Random Sampling**: Selects a specific ratio (e.g., 5%) of files.
3.  **Minimum Guarantee**: Ensures **at least one file** is selected for every dataset type defined, even if the ratio yields < 1.
4.  **State Management**: Automatically generates a `history.yaml` to track moved files.
    *   If `history.yaml` exists, the script **replays** the move operation (idempotent) instead of re-sampling.
5.  **Reset Capability**: Can move all sampled files back to the origin directory using the history file.
6.  **Interactive Safety**: Prompts the user before overwriting existing files in target directories.
7.  **Merging Support**: Integrates with `haddnano.py` to merge the sampled files into a single `.root` file per category.

## Prerequisites

*   Python 3.x
*   PyYAML library
*   (Optional) `haddnano.py` script if you intend to use the merge function.

```bash
pip install pyyaml
```

## Configuration

Create a `config.yaml` file to define your dataset categories and the file prefixes belonging to them.

**Example `config.yaml`:**

```yaml
ttbar-powheg:
  - TTToSemiLeptonic_Vcb_TuneCP5_13TeV-powheg-pythia8

single-top:
  - ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8
  - ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8

tt-had:
  - TTToHadronic_TuneCP5_13TeV-powheg-pythia8
```

## Usage

### 1. Random Sampling (First Run)
Randomly select 5% of files from `--origin`, move them to `--train`, and save the record to `history.yaml`.

```bash
python sampler.py \
  --config config.yaml \
  --origin /data/storage/pool \
  --train /data/work/train_sample \
  --ratio 0.05
```

### 2. Sampling + Merging
Move the files **and** immediately merge them using `haddnano.py`.
The output files will be named `{category}_train.root` (e.g., `ttbar-powheg_train.root`) inside the `--merge_out` directory.

```bash
python sampler.py \
  --config config.yaml \
  --origin /data/storage/pool \
  --train /data/work/train_sample \
  --ratio 0.05 \
  --merge_out /data/work/merged_files \
  --haddnano ./haddnano.py
```

### 3. Replay (Existing History)
If you run the command again while `history.yaml` exists, the script ignores `--ratio`. It checks the history file and ensures those specific files are in the `--train` directory.

```bash
python sampler.py \
  --config config.yaml \
  --origin /data/storage/pool \
  --train /data/work/train_sample
```

### 4. Reset (Restore Files)
Move all files listed in `history.yaml` from `--train` back to `--origin` and delete the history file.

```bash
python sampler.py \
  --config config.yaml \
  --origin /data/storage/pool \
  --train /data/work/train_sample \
  --reset
```

## Arguments

| Argument | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `--config` | Yes | - | Path to the dataset configuration YAML file. |
| `--origin` | Yes | - | Source directory containing the original ROOT files. |
| `--train` | Yes | - | Target directory where sampled files will be moved. |
| `--ratio` | No | `0.1` | Sampling ratio (0.0 to 1.0). Default is 10%. |
| `--history`| No | `history.yaml` | Path to save/read the history file. |
| `--reset` | No | `False` | Flag to move files back to origin and clear history. |
| `--merge_out`| No | `None` | Directory to save merged ROOT files. If omitted, merging is skipped. |
| `--haddnano` | No | `haddnano.py` | Path to the external `haddnano.py` script. |

## Logic Details

1.  **Selection**: usage is calculated as `max(1, count * ratio)`.
2.  **Merging**: The script constructs the following command:
    ```bash
    python {haddnano_path} {merge_out}/{category}_train.root {file1} {file2} ...
    ```
3.  **Collision**: If a file with the same name exists in the destination (during move or merge), the script pauses and asks for confirmation via the terminal (`y/n`).

# NanoAOD File Tagger

A flexible post-processing tool for CMS NanoAOD samples. This script automatically adds new branches (flags/tags) to your ROOT files based on their filenames using a simple YAML configuration.

It is built on top of `PhysicsTools.NanoAODTools`.

## Features

- **Pattern Matching**: Categorizes files by matching substrings in their filenames (e.g., "QCD", "TTToSemiLeptonic").
- **Dynamic Branch Creation**: Creates integer, float, or boolean branches.
- **Efficient**: Calculates the value **once per file** (in `beginFile`) and fills it efficiently for every event, avoiding slow string matching inside the event loop.
- **Batch Processing**: Can process a single `.root` file or an entire directory of files.

## Prerequisites

- Python 3
- [NanoAOD-tools](https://github.com/cms-nanoAOD/nanoAOD-tools) installed and available in your `PYTHONPATH`.
- `PyYAML` (`pip install pyyaml`)

## Configuration (`config.yaml`)

The behavior of the tagger is controlled entirely by a YAML file. It has two main sections: `file_groups` and `branches`.

### 1. `file_groups`
Define categories (groups) by listing substrings that appear in the filenames.

```yaml
file_groups:
  # Group Name: [List of substrings to match]
  qcd_background:
    - "QCD_HT"
    - "QCD_Pt"
  
  top_process:
    - "TTToSemiLeptonic"
    - "TTTo2L2Nu"
```

### 2. `branches`
Define the new branches to create and map the groups to specific values.

```yaml
branches:
  # Name of the new branch
  is_qcd:
    type: "I"        # 'I'=Int, 'F'=Float, 'O'=Bool
    title: "Flag: Is this event QCD?"
    default: 0       # Value used if file matches NO groups
    mappings:
      # Group Name : Value to fill
      qcd_background: 1
      top_process: 0

  # You can add multiple branches
  process_id:
    type: "I"
    default: -1
    mappings:
      qcd_background: 100
      top_process: 200
```

## Usage

Save the python script as `tagger.py`.

### Command Line Arguments

- `-i`, `--input`: Path to a single `.root` file OR a directory containing root files.
- `-o`, `--output`: Output directory where processed files will be saved.
- `-c`, `--config`: Path to the YAML configuration file.

### Example Commands

**Process a single file:**
```bash
python3 tagger.py \
  -i /data/input/TTToSemiLeptonic_TuneCP5.root \
  -o /data/output/ \
  -c config.yaml
```

**Process an entire directory:**
```bash
python3 tagger.py \
  -i /data/input_samples/ \
  -o /data/output_samples/ \
  -c config.yaml
```

## Logic Flow

1. **Startup**: The script loads the YAML config.
2. **File Loop**: It iterates through every `.root` file found in the input.
3. **Categorization (`beginFile`)**:
   - It looks at the filename (e.g., `QCD_HT200.root`).
   - It checks the `file_groups` in YAML.
   - It determines that this file belongs to `qcd_background`.
   - It looks at the `branches` config. For `is_qcd`, it sees `qcd_background` maps to `1`.
4. **Event Loop (`analyze`)**:
   - For every event in the tree, it fills the `is_qcd` branch with `1`.
5. **Output**: The modified tree is written to the output directory.

## Troubleshooting

- **Warning: Source branch not found**: This script does not read old branches, so you won't see this error. It only creates *new* constant branches.
- **Value is always default**: Check your `file_groups` substrings. The matching is case-sensitive. Ensure the filename actually contains the string you defined.


# NanoAOD Branch Renamer

A flexible, configuration-driven tool based on **NanoAODTools** to rename, copy, and restructure branches in ROOT files.

It allows you to define new branches via a YAML file, supports both **Direct Copying** (fast) and **Object-Oriented Extraction** (flexible), and automatically handles the removal of old branches.

## Features

*   **YAML Configuration**: Define all logic in a simple text file.
*   **Two Modes**:
    1.  **Direct Copy**: Fast `getattr` from one branch to another.
    2.  **Collection Loop**: Pythonic access (e.g., `[j.pt for j in jets]`), useful for complex logic.
*   **Auto-Drop**: Automatically generates the `keep/drop` rules to remove old branches in the output.
*   **Vector Support**: Correctly handles array lengths (`lenVar`).

## Prerequisites

*   Python 3
*   ROOT
*   [NanoAODTools](https://github.com/cms-nanoAOD/nanoAOD-tools) (Standalone or CMSSW)

## Usage

```bash
python rename_objects.py -i <INPUT_FILE> -o <OUTPUT_DIR> -c <CONFIG_YAML>
```

### Arguments
*   `-i`, `--input`: Path to the input ROOT file.
*   `-o`, `--output`: Directory where the output file will be saved.
*   `-c`, `--config`: Path to the YAML configuration file.

## Configuration (`config.yaml`)

The YAML file controls the entire process. It has two main sections: `creates` and `drops`.

### 1. `creates`
Defines the new branches. You can create a branch using **Source Mode** (Raw copy) or **Collection Mode** (Object attribute).

| Key | Description | Required |
| :--- | :--- | :--- |
| `type` | ROOT type char (`F`=Float, `I`=Int, `O`=Bool, etc.) | Yes |
| `lenVar` | Name of the **new** length branch (for arrays/vectors) | No (Scalar) |
| `src` | **Mode A**: The exact name of the old branch to copy. | Conditional |
| `collection`| **Mode B**: The collection name (e.g., `ak8jet`). | Conditional |
| `attribute` | **Mode B**: The attribute to extract (e.g., `pt`). | Conditional |

### 2. `drops`
A list of old branch names (or patterns) to remove from the output file.

## Example Configuration

**Goal**: Rename `n_ak8jet` to `n_ak8`, and rename `ak8jet_pt` to `ak8_pt`.

```yaml
creates:
  # 1. Rename the Counter (Scalar)
  # Method: Direct Copy
  n_ak8:
    src: n_ak8jet
    type: I

  # 2. Rename a Vector Variable
  # Method: Collection Loop (reads object properties)
  # Logic equivalent to: [obj.pt for obj in Collection(event, "ak8jet")]
  ak8_pt:
    collection: ak8jet
    attribute: pt
    type: F
    lenVar: n_ak8  # Points to the NEW counter defined above

  # 3. Rename another Vector
  # Method: Direct Copy (Faster, reads raw branch)
  ak8_eta:
    src: ak8jet_eta
    type: F
    lenVar: n_ak8

drops:
  # Remove the old data to save space
  - n_ak8jet
  - ak8jet_* 
```

## How it Works

1.  **Init**: The script reads the YAML and books new branches in the output tree.
2.  **Analyze**:
    *   For `collection` mode: It initializes a `Collection` object (e.g., `ak8jets`) and iterates over it to fill the list.
    *   For `src` mode: It directly reads the branch value.
3.  **Write**: It generates a temporary "drop file" based on your `drops` list and passes it to the `PostProcessor`. The output file will contain the new branches and exclude the dropped ones.

## Example Run

```bash
# 1. Create config
vim rename.yaml

# 2. Run script
python rename_objects.py -i input.root -o output/ -c rename.yaml

# 3. Result
# output/input_Skim.root will be created.
```