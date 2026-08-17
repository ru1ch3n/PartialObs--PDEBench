(function () {
  "use strict";

  const form = document.getElementById("benchmark-builder");
  if (!form) return;

  const byId = (id) => document.getElementById(id);
  const selectors = {
    environment: byId("builder-environment"),
    tier: byId("builder-tier"),
    pde: byId("builder-pde"),
    boundary: byId("builder-boundary"),
    setting: byId("builder-setting"),
    regime: byId("builder-regime"),
    task: byId("builder-task"),
    mask: byId("builder-mask"),
    split: byId("builder-split"),
    model: byId("builder-model"),
    quality: byId("builder-quality"),
  };
  const inputs = {
    threshold: byId("builder-threshold"),
    name: byId("builder-name"),
    dataRoot: byId("builder-data-root"),
    runRoot: byId("builder-run-root"),
    workers: byId("builder-workers"),
    shardSize: byId("builder-shard-size"),
    group: byId("builder-group"),
  };
  const campaignControls = {
    preset: byId("campaign-preset"),
    gpus: byId("campaign-gpus"),
    days: byId("campaign-days"),
    utilizationLow: byId("campaign-utilization-low"),
    utilizationHigh: byId("campaign-utilization-high"),
  };
  const queryKeys = {
    environment: "env",
    tier: "tier",
    pde: "pde",
    boundary: "boundary",
    setting: "setting",
    regime: "regime",
    task: "task",
    mask: "mask",
    split: "split",
    model: "model",
    quality: "quality",
  };

  let options = null;
  let codeByTab = {};
  let activeTab = "setup";

  function optionRows(kind) {
    if (kind === "campaign_presets") return options.campaign_planner.presets || [];
    return options[kind] || [];
  }

  function findRow(kind, value) {
    return optionRows(kind).find((row) => row.value === value);
  }

  function populateSelect(select) {
    const kind = select.dataset.option;
    const allLabel = select.dataset.allLabel;
    select.replaceChildren();
    if (allLabel) {
      const all = document.createElement("option");
      all.value = "all";
      all.textContent = allLabel;
      select.appendChild(all);
    }
    optionRows(kind).forEach((row) => {
      const item = document.createElement("option");
      item.value = row.value;
      item.textContent = row.label;
      select.appendChild(item);
    });
    const preferred = select.dataset.default || "";
    if (Array.from(select.options).some((item) => item.value === preferred)) {
      select.value = preferred;
    }
  }

  function applyDeepLink() {
    const params = new URLSearchParams(window.location.search);
    Object.entries(queryKeys).forEach(([key, queryKey]) => {
      const value = params.get(queryKey);
      const select = selectors[key];
      if (value && Array.from(select.options).some((item) => item.value === value)) {
        select.value = value;
      }
    });
    const threshold = params.get("limit");
    if (threshold !== null && Number.isFinite(Number(threshold)) && Number(threshold) >= 0) {
      inputs.threshold.value = threshold;
    }
    const workers = params.get("workers");
    if (workers !== null && Number.isInteger(Number(workers))) inputs.workers.value = workers;
    const shardSize = params.get("shard");
    if (shardSize !== null && Number.isInteger(Number(shardSize))) {
      inputs.shardSize.value = shardSize;
    }
    const campaignPreset = params.get("campaign");
    if (
      campaignPreset &&
      Array.from(campaignControls.preset.options).some((item) => item.value === campaignPreset)
    ) {
      campaignControls.preset.value = campaignPreset;
    }
    [
      ["gpus", campaignControls.gpus],
      ["days", campaignControls.days],
      ["util_low", campaignControls.utilizationLow],
      ["util_high", campaignControls.utilizationHigh],
    ].forEach(([key, input]) => {
      const value = params.get(key);
      if (value !== null && Number.isFinite(Number(value))) input.value = value;
    });
  }

  function updateDeepLink(state, campaignState) {
    const params = new URLSearchParams();
    Object.entries(queryKeys).forEach(([key, queryKey]) => {
      params.set(queryKey, state[key]);
    });
    if (state.threshold !== null) params.set("limit", String(state.threshold));
    params.set("workers", String(state.workers));
    params.set("shard", String(state.shardSize));
    params.set("campaign", campaignState.preset);
    params.set("gpus", String(campaignState.gpus));
    params.set("days", String(campaignState.days));
    params.set("util_low", String(campaignState.utilizationLow));
    params.set("util_high", String(campaignState.utilizationHigh));
    const query = params.toString();
    window.history.replaceState(null, "", `${window.location.pathname}?${query}`);
  }

  function integerValue(input, fallback, minimum, maximum) {
    const value = Number.parseInt(input.value, 10);
    if (!Number.isFinite(value)) return fallback;
    return Math.min(maximum, Math.max(minimum, value));
  }

  function cleanName(value) {
    const cleaned = String(value || "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "");
    return cleaned || "pdeobs-custom";
  }

  function readState() {
    const rawThreshold = inputs.threshold.value.trim();
    const parsedThreshold = rawThreshold === "" ? null : Number(rawThreshold);
    return {
      environment: selectors.environment.value,
      tier: selectors.tier.value,
      pde: selectors.pde.value,
      boundary: selectors.boundary.value,
      setting: selectors.setting.value,
      regime: selectors.regime.value,
      task: selectors.task.value,
      mask: selectors.mask.value,
      split: selectors.split.value,
      model: selectors.model.value,
      quality: selectors.quality.value,
      threshold:
        Number.isFinite(parsedThreshold) && parsedThreshold >= 0 ? parsedThreshold : null,
      name: cleanName(inputs.name.value),
      dataRoot: inputs.dataRoot.value.trim() || "datasets",
      runRoot: inputs.runRoot.value.trim() || "runs",
      workers: integerValue(inputs.workers, 4, 1, 128),
      shardSize: integerValue(inputs.shardSize, 700, 1, 2000),
      group: inputs.group.value.trim() || "YOUR_GROUP",
    };
  }

  function readCampaignState() {
    const utilizationA = integerValue(campaignControls.utilizationLow, 75, 1, 100);
    const utilizationB = integerValue(campaignControls.utilizationHigh, 80, 1, 100);
    return {
      preset: campaignControls.preset.value,
      gpus: integerValue(campaignControls.gpus, 12, 1, 512),
      days: integerValue(campaignControls.days, 10, 1, 365),
      utilizationLow: Math.min(utilizationA, utilizationB),
      utilizationHigh: Math.max(utilizationA, utilizationB),
    };
  }

  function selectedCampaignPreset(campaignState) {
    return optionRows("campaign_presets").find(
      (preset) => preset.value === campaignState.preset,
    );
  }

  function selectedValues(kind, value) {
    return value === "all" ? optionRows(kind).map((row) => row.value) : [value];
  }

  function yamlString(value) {
    return JSON.stringify(String(value));
  }

  function yamlList(values) {
    return `[${values.map(yamlString).join(", ")}]`;
  }

  function shellQuote(value) {
    return `'${String(value).replace(/'/g, `'"'"'`)}'`;
  }

  function lineCommand(command, args) {
    if (!args.length) return command;
    return `${command} \\\n  ${args.join(" \\\n  ")}`;
  }

  function qualityArgs(state, aggregate) {
    const args = [];
    if (state.quality === "strict" || state.quality === "publication") {
      args.push("--quality-strict");
    }
    if (state.threshold !== null) {
      args.push("--max-pde-loss", String(state.threshold));
    }
    if (state.quality === "publication") {
      args.push("--require-all-pdes", "--require-validated-solvers");
    }
    if (!aggregate) return args;
    return args;
  }

  function buildYaml(state) {
    const pdes = selectedValues("pdes", state.pde);
    const boundaries = selectedValues("boundaries", state.boundary);
    const settings = selectedValues("settings", state.setting);
    const regimes = selectedValues("regimes", state.regime);
    const threshold = state.threshold === null ? "null" : String(state.threshold);
    return [
      "# Generated by the PDE-OBS Benchmark Builder.",
      "# Keep this file beside default.yaml, copied from configs/dataset/default.yaml.",
      "# Record the exact Git revision used for generation.",
      "include: default.yaml",
      `name: ${yamlString(state.name)}`,
      `tier: ${state.tier}`,
      `families: ${yamlList(pdes)}`,
      `boundaries: ${yamlList(boundaries)}`,
      `settings: ${yamlList(settings)}`,
      `regimes: ${yamlList(regimes)}`,
      `shard_size: ${state.shardSize}`,
      "quality:",
      "  enabled: true",
      `  profile: ${state.quality}`,
      "  require_pde_loss: true",
      "  thresholds:",
      "    finite_fraction_min: 1.0",
      "    geometry_binary_max_error_max: 1.0e-6",
      "    initial_condition_loss_normalized_max: 1.0e-6",
      "    boundary_condition_loss_normalized_max: 1.0e-4",
      `    pde_loss_normalized_max: ${threshold}`,
      "    divergence_loss_normalized_max: null",
      "output:",
      '  root: "${PDEOBS_DATA:-datasets}"',
      "  atomic: true",
      "  checksum: sha256",
    ].join("\n");
  }

  function campaignDatasetRow(preset) {
    return options.campaign_planner.dataset_accounting[preset.tier];
  }

  function buildCampaignManifest(campaignState) {
    const preset = selectedCampaignPreset(campaignState);
    const dataset = campaignDatasetRow(preset);
    const methods = options.protocol_methods.filter((row) =>
      preset.methods.includes(row.value),
    );
    const blocked = methods.filter((row) => row.command_generation === "blocked");
    const trainingPerPde =
      dataset.training_records_per_pde ??
      dataset.training_records_per_pde_approximately;
    const methodPlan = methods.flatMap((row) => [
      `  - method_id: ${yamlString(row.value)}`,
      `    observations: ${yamlList(preset.method_observations[row.value] || [])}`,
      `    seeds: ${row.default_seeds}`,
      `    fit_scope: ${yamlString(row.fit_scope)}`,
      `    execution_status: ${yamlString(row.execution_status)}`,
      `    command_generation: ${yamlString(row.command_generation)}`,
    ]);
    return [
      "# PLANNING MANIFEST ONLY - NOT A PDE-OBS CLI CONFIG.",
      "# No training, evaluation, or scheduler command is emitted from this tab.",
      "# Add and validate every missing adapter before creating an executable campaign.",
      `schema_version: ${options.campaign_planner.schema_version}`,
      `protocol_document: ${yamlString(options.campaign_planner.protocol_url)}`,
      `preset: ${yamlString(preset.value)}`,
      `tier: ${preset.tier}`,
      "primary_comparison:",
      "  name: matched_mask_iid",
      "  training_mask_equals_evaluation_mask: true",
      "  independent_checkpoint_per_pde_and_observation: true",
      "secondary_comparison:",
      "  name: random_3pct_mask_transfer",
      "  training_mask: random_3pct",
      "  replaces_primary: false",
      `methods: ${yamlList(preset.methods)}`,
      `blocked_methods: ${yamlList(blocked.map((row) => row.value))}`,
      "method_plan:",
      ...methodPlan,
      "data_accounting:",
      `  total_records: ${dataset.total_records}`,
      `  records_per_pde: ${dataset.records_per_pde}`,
      `  training_records_per_pde: ${trainingPerPde}`,
      "campaign_accounting:",
      `  result_cells: ${preset.result_cells}`,
      `  neural_training_jobs: ${preset.neural_training_jobs}`,
      `  pod_fit_jobs: ${preset.pod_fit_jobs}`,
      `  prior_preparation_jobs: [${preset.prior_preparation_jobs_min}, ${preset.prior_preparation_jobs_max}]`,
      `  total_preparation_jobs: [${preset.preparation_jobs_min}, ${preset.preparation_jobs_max}]`,
      `  raw_evaluation_runs: ${preset.raw_evaluation_runs}`,
      "budget:",
      "  status: unmeasured_planning_estimate",
      `  dedicated_gpus: ${campaignState.gpus}`,
      `  campaign_days: ${campaignState.days}`,
      `  utilization_fraction: [${campaignState.utilizationLow / 100}, ${campaignState.utilizationHigh / 100}]`,
      "  source_hardware: NVIDIA RTX A6000",
      "  seawulf_a100_transferable: false",
      "execution:",
      "  runnable: false",
      "  reason: planning rows without registered adapters are present",
    ].join("\n");
  }

  function publicationBlock(state) {
    const reasons = [];
    if (state.quality === "publication" && state.pde !== "all") {
      reasons.push("select all seven PDE families");
    }
    if (state.quality === "publication" && state.threshold === null) {
      reasons.push("enter a frozen, resolution-specific calibrated PDE-loss limit");
    }
    if (state.quality === "publication") {
      reasons.push("use an explicit expert config only after an external trusted verifier exists for solver evidence and the per-stratum threshold table");
      reasons.push("use a complete bounded Navier-Stokes residual contract; velocity-only storage is currently partial");
    }
    if (!reasons.length) return "";
    return [
      "# Publication-candidate generation is intentionally blocked in the web Builder.",
      `echo ${shellQuote(`Publication-candidate quality gate: ${reasons.join("; ")}.`)} >&2`,
      "exit 2",
      "",
    ].join("\n");
  }

  function buildSetup(state) {
    const common = [
      "git clone https://github.com/ru1ch3n/PartialObs--PDEBench.git",
      "cd PartialObs--PDEBench",
    ];
    if (state.environment === "seawulf") {
      return [
        "# Run this command separately on your local computer:",
        "# ssh YOUR_NETID@milan.seawulf.stonybrook.edu",
        "# Then paste the remaining commands only after the SeaWulf prompt appears.",
        "set -Eeuo pipefail",
        "module load slurm",
        ...common,
        "git checkout YOUR_RELEASE_TAG_OR_COMMIT",
        `export PDEOBS_GROUP=${shellQuote(state.group)}`,
        'export PDEOBS_COMMIT="$(git rev-parse --short=12 HEAD)"',
        'export PDEOBS_ENV="/gpfs/projects/$PDEOBS_GROUP/envs/pdeobs-$PDEOBS_COMMIT"',
        `export PDEOBS_DATA=${shellQuote(state.dataRoot)}`,
        `export PDEOBS_RUNS=${shellQuote(state.runRoot)}`,
        'mkdir -p logs "$PDEOBS_DATA" "$PDEOBS_RUNS"',
        "",
        "# Build only inside a compute allocation, never on the login node.",
        "srun --partition=short-40core-shared --nodes=1 --ntasks=1 \\",
        "  --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash -l",
        "bash hpc/seawulf/bootstrap.sh",
        "exit",
        '"$PDEOBS_ENV/bin/python" -m pdeobs doctor --cluster seawulf --offline',
      ].join("\n");
    }
    const session = state.environment === "server" ? ["tmux new -s pdeobs"] : [];
    const install =
      state.environment === "server"
        ? 'python -m pip install ".[train,test]"'
        : 'python -m pip install -e ".[train,test]"';
    return [
      "set -Eeuo pipefail",
      ...common,
      ...(state.environment === "server" ? ["git checkout YOUR_RELEASE_TAG_OR_COMMIT"] : []),
      "python3 -m venv .venv",
      "source .venv/bin/activate",
      "python -m pip install --upgrade pip setuptools wheel",
      install,
      `export PDEOBS_DATA=${shellQuote(state.dataRoot)}`,
      `export PDEOBS_RUNS=${shellQuote(state.runRoot)}`,
      'mkdir -p "$PDEOBS_DATA" "$PDEOBS_RUNS"',
      "pdeobs doctor",
      "pdeobs protocol --check",
      ...session,
    ].join("\n");
  }

  function trainingSuppression(state) {
    const task = findRow("tasks", state.task);
    const model = findRow("models", state.model);
    const taskCapability = {
      sparse_recovery: "recovery",
      forward_prediction: "forward",
      inverse_prediction: "inverse",
      semantic_retrieval: "retrieval",
      world_modeling: "rollout",
      solver_routing: "routing",
      foundation_transfer: "pretraining",
    }[state.task];
    if (!task || task.status !== "executable_field_task") {
      return `${task ? task.label : state.task} is a protocol/API anchor, not a field Trainer task`;
    }
    if (!model || model.capabilities_known !== true) {
      return `${model ? model.label : state.model} has no machine-readable training capabilities`;
    }
    if (model.trainable !== true) {
      return `${model.label} is a non-trainable baseline and must be evaluated through an explicit benchmark configuration`;
    }
    if (state.model === "autoregressive") {
      return "Autoregressive is a wrapper that needs a concrete one-step base model";
    }
    if (!Array.isArray(model.tasks) || !model.tasks.includes(taskCapability)) {
      return `${model.label} does not advertise ${taskCapability} capability`;
    }
    if (state.split !== "iid") {
      const splitGuidance = {
        boundary_ood: "boundary_ood controls the held-out boundary factor",
        setting_ood: "setting_ood controls the held-out setting factor",
        parameter_ood: "parameter_ood controls the held-out regime factor",
        combination_ood: "combination_ood jointly controls boundary and setting",
        mask_ood: "mask_ood is an evaluation sweep, not one dataset membership",
        time_horizon_ood: "time_horizon_ood is an evaluation sweep, not one dataset membership",
      };
      return `${splitGuidance[state.split] || `${state.split} needs coordinated factors`}; use an explicit experiment YAML`;
    }
    if ([state.pde, state.boundary, state.setting, state.regime].includes("all")) {
      return "factor-sweep training needs one concrete tuple or an explicit multi-factor experiment YAML";
    }
    if (state.task === "inverse_prediction" && state.pde !== "darcy") {
      return "the reference inverse Trainer preset is Darcy-specific";
    }
    if (
      state.task === "world_modeling" &&
      !["heat", "reaction_diffusion", "burgers", "navier_stokes"].includes(state.pde)
    ) {
      return "world modeling requires a temporal PDE family";
    }
    if (
      state.pde === "navier_stokes" &&
      state.boundary !== "periodic"
    ) {
      return "bounded Navier-Stokes stores two-channel velocity and needs an explicit two-channel experiment preset";
    }
    return "";
  }

  function trainingCommand(state) {
    const suppression = trainingSuppression(state);
    if (suppression) {
      return [
        `# Automatic training suppressed: ${suppression}.`,
        "# No one-line training command is generated; use an explicit experiment YAML.",
      ].join("\n");
    }
    const factors = [];
    if (state.pde !== "all") factors.push("--pde", state.pde);
    if (state.boundary !== "all") factors.push("--boundary", state.boundary);
    if (state.setting !== "all") factors.push("--setting", state.setting);
    if (state.regime !== "all") factors.push("--param-regime", state.regime);
    return lineCommand("pdeobs train", [
      "--task",
      state.task,
      "--model",
      state.model,
      "--data",
      '"$DATASET_ROOT"',
      "--split",
      state.split,
      "--mask",
      state.mask,
      ...factors,
      "--output",
      `"$PDEOBS_RUNS/${state.name}-${state.task}-${state.model}"`,
    ]);
  }

  function buildRun(state, yaml) {
    if (state.environment === "seawulf") {
      return "# SeaWulf is selected. Use the 'SeaWulf chain' tab for bounded Slurm arrays, validation, and quality gating.";
    }
    const aggregateArgs = qualityArgs(state, true);
    const auditArgs = qualityArgs(state, false);
    return [
      "set -Eeuo pipefail",
      publicationBlock(state),
      `PDEOBS_DATA_DEFAULT=${shellQuote(state.dataRoot)}`,
      `PDEOBS_RUNS_DEFAULT=${shellQuote(state.runRoot)}`,
      'export PDEOBS_DATA="${PDEOBS_DATA:-$PDEOBS_DATA_DEFAULT}"',
      'export PDEOBS_RUNS="${PDEOBS_RUNS:-$PDEOBS_RUNS_DEFAULT}"',
      ': "${PDEOBS_DATA:?Set PDEOBS_DATA to a writable data directory}"',
      ': "${PDEOBS_RUNS:?Set PDEOBS_RUNS to a writable run directory}"',
      'REPO_ROOT="$(git rev-parse --show-toplevel)"',
      'cd "$REPO_ROOT"',
      'CONFIG_DIR="$PDEOBS_DATA/configs"',
      `CONFIG="$CONFIG_DIR/${state.name}.dataset.yaml"`,
      `DATASET_ROOT="$PDEOBS_DATA/${state.name}"`,
      'PLAN="$DATASET_ROOT/generation-plan.jsonl"',
      'mkdir -p "$CONFIG_DIR" "$DATASET_ROOT" "$PDEOBS_RUNS"',
      'cp configs/dataset/default.yaml "$CONFIG_DIR/default.yaml"',
      'cat > "$CONFIG" <<\'YAML\'',
      yaml,
      "YAML",
      "",
      "pdeobs protocol --check",
      lineCommand("pdeobs plan", ["--config", '"$CONFIG"', "--output", '"$PLAN"']),
      lineCommand("pdeobs generate", [
        "--config",
        '"$CONFIG"',
        "--output",
        '"$DATASET_ROOT"',
        "--plan",
        '"$PLAN"',
        "--num-workers",
        String(state.workers),
      ]),
      lineCommand("pdeobs aggregate", [
        "--input",
        '"$DATASET_ROOT"',
        "--output",
        '"$DATASET_ROOT/summary.json"',
        "--validate-shards",
        "--expected-plan",
        '"$PLAN"',
        ...aggregateArgs,
      ]),
      lineCommand("pdeobs quality", [
        "--input",
        '"$DATASET_ROOT"',
        "--output",
        '"$DATASET_ROOT/quality-report.json"',
        ...auditArgs,
      ]),
      "",
      trainingCommand(state),
    ].join("\n");
  }

  function buildSeaWulf(state, yaml) {
    const aggregateArgs = qualityArgs(state, true);
    return [
      "set -Eeuo pipefail",
      publicationBlock(state),
      "# Run after the SeaWulf setup tab. This submits exactly one window of at most 100 tasks.",
      ': "${PDEOBS_ENV:?Run the SeaWulf setup tab and export PDEOBS_ENV}"',
      ': "${PDEOBS_DATA:?Run the SeaWulf setup tab and export PDEOBS_DATA}"',
      ': "${PDEOBS_RUNS:?Run the SeaWulf setup tab and export PDEOBS_RUNS}"',
      'REPO_ROOT="$(git rev-parse --show-toplevel)"',
      'cd "$REPO_ROOT"',
      'CONFIG_DIR="$PDEOBS_DATA/configs"',
      `CONFIG="$CONFIG_DIR/${state.name}.dataset.yaml"`,
      `DATASET_ROOT="$PDEOBS_DATA/${state.name}"`,
      'PLAN="$DATASET_ROOT/generation-plan.jsonl"',
      'mkdir -p "$CONFIG_DIR" "$DATASET_ROOT" "$PDEOBS_RUNS" logs',
      'cp configs/dataset/default.yaml "$CONFIG_DIR/default.yaml"',
      'cat > "$CONFIG" <<\'YAML\'',
      yaml,
      "YAML",
      "",
      '"$PDEOBS_ENV/bin/python" -m pdeobs protocol --check',
      lineCommand('"$PDEOBS_ENV/bin/python" -m pdeobs plan', [
        "--config",
        '"$CONFIG"',
        "--output",
        '"$PLAN"',
      ]),
      'task_count="$(awk \'NF { count++ } END { print count+0 }\' "$PLAN")"',
      '(( task_count > 0 )) || { echo "Generation plan is empty" >&2; exit 2; }',
      'start="${PDEOBS_WINDOW_START:-0}"',
      '[[ "$start" =~ ^[0-9]+$ ]] || { echo "PDEOBS_WINDOW_START must be a non-negative integer" >&2; exit 2; }',
      '(( start < task_count )) || { echo "Window start $start is outside 0-$((task_count - 1))" >&2; exit 2; }',
      'stop=$((start + 99))',
      '(( stop < task_count )) || stop=$((task_count - 1))',
      'submission="$(bash hpc/seawulf/submit_generation.sh "$PLAN" "$CONFIG" "$DATASET_ROOT" "$start" "$stop")"',
      'generation_job="${submission##* }"',
      'generation_job="${generation_job%%;*}"',
      '[[ "$generation_job" =~ ^[0-9]+$ ]] || { echo "Could not parse Slurm job ID: $submission" >&2; exit 2; }',
      'echo "generation window $start-$stop: $generation_job"',
      "",
      'if (( stop + 1 < task_count )); then',
      '  echo "Wait for job $generation_job to complete successfully."',
      '  echo "Then export PDEOBS_WINDOW_START=$((stop + 1)) and rerun this SeaWulf tab."',
      '  squeue -j "$generation_job"',
      'else',
      '  validation_job="$(sbatch --parsable --dependency="afterok:$generation_job" hpc/seawulf/aggregate_cpu.sbatch "$DATASET_ROOT" "$DATASET_ROOT/summary.json" "$PLAN"' +
        (aggregateArgs.length ? " " + aggregateArgs.join(" ") : "") +
        ')"',
      '  validation_job="${validation_job%%;*}"',
      '  echo "validation + quality gate: $validation_job"',
      '  squeue -j "$generation_job,$validation_job"',
      'fi',
      "",
      "# Inspect summary.json, summary.quality.json/.csv, per-shard *.quality.json,",
      "# any strict *.quality-failures.jsonl records, logs, and checksums before training.",
      "# SeaWulf scratch is temporary and not backed up; archive validated outputs independently.",
    ].join("\n");
  }

  function addListItems(container, values) {
    container.replaceChildren();
    values.forEach((value) => {
      const item = document.createElement("li");
      item.textContent = value;
      container.appendChild(item);
    });
  }

  function renderQualityTable() {
    const body = byId("builder-quality-body");
    body.replaceChildren();
    options.pdes.forEach((pde) => {
      const row = document.createElement("tr");
      [pde.label, pde.loss, pde.note].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
    addListItems(byId("builder-quality-outputs"), options.quality_outputs);
  }

  function formattedRange(low, high) {
    if (low === high) return Number(low).toLocaleString();
    return `${Number(low).toLocaleString()}-${Number(high).toLocaleString()}`;
  }

  function methodCampaignCounts(method, observationCount) {
    const pdeCount = Number(options.campaign_planner.pde_count);
    const seeds = Number(method.default_seeds);
    const cells = pdeCount * observationCount;
    if (method.fit_scope === "once_per_pde_and_observation") {
      return { preparation: cells * seeds, evaluations: cells * seeds };
    }
    if (method.fit_scope === "once_per_pde_training_split") {
      return { preparation: pdeCount, evaluations: cells };
    }
    if (method.fit_scope === "once_per_pde_prior") {
      return { preparation: `0-${pdeCount}`, evaluations: cells };
    }
    return { preparation: 0, evaluations: cells };
  }

  function renderCampaignMethodTable(preset) {
    const body = byId("campaign-method-body");
    body.replaceChildren();
    options.protocol_methods.forEach((method) => {
      const selectedMasks = preset.method_observations[method.value] || [];
      const row = document.createElement("tr");
      if (!selectedMasks.length) row.classList.add("campaign-method-omitted");
      const counts = methodCampaignCounts(method, selectedMasks.length);
      const status = selectedMasks.length
        ? method.builder_available
          ? "Single-run adapter present; campaign generation remains planning-only"
          : "Blocked: no registered PDE-OBS adapter"
        : "Not included in this preset";
      const values = [
        method.label,
        method.fit_scope.replaceAll("_", " "),
        String(method.default_seeds),
        selectedMasks.length
          ? `${selectedMasks.length}: ${selectedMasks.join(", ")}`
          : "-",
        selectedMasks.length ? String(counts.preparation) : "-",
        selectedMasks.length ? String(counts.evaluations) : "-",
        status,
      ];
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
  }

  function renderCampaign(campaignState) {
    const preset = selectedCampaignPreset(campaignState);
    const dataset = campaignDatasetRow(preset);
    const trainingPerPde =
      dataset.training_records_per_pde ??
      dataset.training_records_per_pde_approximately;
    const theoretical = campaignState.gpus * campaignState.days * 24;
    const safeLow = theoretical * campaignState.utilizationLow / 100;
    const safeHigh = theoretical * campaignState.utilizationHigh / 100;
    byId("campaign-data-pool").textContent = Number(dataset.total_records).toLocaleString();
    byId("campaign-data-per-pde").textContent = Number(
      dataset.records_per_pde,
    ).toLocaleString();
    byId("campaign-train-per-pde").textContent = Number(trainingPerPde).toLocaleString();
    byId("campaign-result-cells").textContent = Number(
      preset.result_cells,
    ).toLocaleString();
    byId("campaign-preparation-jobs").textContent = formattedRange(
      preset.preparation_jobs_min,
      preset.preparation_jobs_max,
    );
    byId("campaign-evaluation-runs").textContent = Number(
      preset.raw_evaluation_runs,
    ).toLocaleString();
    byId("campaign-safe-capacity").textContent = `${formattedRange(
      Math.round(safeLow),
      Math.round(safeHigh),
    )} GPU-h`;

    let feasibility = "Pilot required: this preset has no defensible GPU-hour estimate.";
    if (preset.gpu_hours_low !== null && preset.gpu_hours_high !== null) {
      byId("campaign-gpu-estimate").textContent = `${formattedRange(
        preset.gpu_hours_low,
        preset.gpu_hours_high,
      )} GPU-h`;
      if (preset.gpu_hours_low > safeHigh) {
        feasibility = "Not feasible within the selected safe planning capacity.";
      } else if (preset.gpu_hours_high > safeLow) {
        feasibility = "Feasible only as a tight planning overlap; pilot and monitor first.";
      } else {
        feasibility = "Fits the selected planning capacity, subject to a measured pilot.";
      }
    } else {
      byId("campaign-gpu-estimate").textContent = "Pilot required";
    }
    byId("campaign-feasibility").textContent = feasibility;
    byId("campaign-recommendation").textContent = preset.recommendation;

    const included = options.protocol_methods.filter((method) =>
      preset.methods.includes(method.value),
    );
    const blocked = included.filter((method) => method.command_generation === "blocked");
    const messages = [
      "GPU-hour figures are unmeasured A6000 planning estimates, not benchmark results.",
      "SeaWulf uses a shared A100 queue; run a SeaWulf pilot and do not transfer the A6000 estimate.",
      "Primary results use matched-mask training; random 3% transfer is a separate mask-OOD analysis.",
      `Campaign commands are blocked for: ${blocked.map((method) => method.label).join(", ")}.`,
    ];
    if (preset.tier === "full") {
      messages.push("The full tier contains 80,000 records per PDE, but only 56,000 belong to the canonical training split.");
    }
    addListItems(byId("campaign-warnings"), messages);
    renderCampaignMethodTable(preset);
  }

  function warningMessages(state, plan) {
    const messages = [];
    const suppression = trainingSuppression(state);
    if (suppression) messages.push(`Automatic training is suppressed: ${suppression}.`);
    if (state.quality === "report") {
      messages.push("Report mode measures quality but does not reject a shard solely for an uncalibrated PDE loss.");
    }
    if (state.quality === "strict" && state.threshold === null) {
      messages.push("Strict mode gates finite/geometry/IC/BC checks; enter a calibrated PDE-loss limit to gate the residual too.");
    }
    if (state.quality === "strict" || state.quality === "publication") {
      messages.push("Rejected strict-generation samples are recorded in *.quality-failures.jsonl for audit and retry.");
    }
    if (state.quality === "publication") {
      if (state.pde !== "all") messages.push("Publication-candidate mode requires all seven PDE families.");
      if (state.threshold === null) messages.push("Publication-candidate mode requires a frozen, family/boundary/resolution-calibrated PDE-loss limit.");
      messages.push("The Builder will not submit publication-candidate generation. Use an explicit expert config only after validated solver/evidence, a per-stratum threshold table, and a complete bounded Navier-Stokes residual contract exist.");
      messages.push("A validated plugin alone is insufficient for bounded Navier-Stokes while velocity-only storage permits only a partial residual diagnostic.");
      messages.push("Even after this candidate gate passes, publication_ready remains false until the canonical full-factor expected plan/checksums and independent release evidence pass.");
    }
    if (state.tier === "full") {
      messages.push("The full matrix is a large campaign. Complete smoke and signal tiers, estimate storage/runtime, and archive validated results first.");
    }
    if (plan.jobs > 100 && state.environment === "seawulf") {
      messages.push("The SeaWulf code submits one window of at most 100 array tasks. After it succeeds, rerun with the displayed PDEOBS_WINDOW_START value.");
    }
    return messages;
  }

  function calculatePlan(state) {
    const tier = findRow("tiers", state.tier);
    const pdeCount = selectedValues("pdes", state.pde).length;
    const boundaryCount = selectedValues("boundaries", state.boundary).length;
    const settingCount = selectedValues("settings", state.setting).length;
    const regimes = selectedValues("regimes", state.regime);
    const macroCases = pdeCount * boundaryCount * settingCount;
    const samplesPerMacro = regimes.reduce(
      (total, regime) => total + Number(tier.regime_counts[regime] || 0),
      0,
    );
    const jobsPerMacro = regimes.reduce(
      (total, regime) =>
        total + Math.ceil(Number(tier.regime_counts[regime] || 0) / state.shardSize),
      0,
    );
    return {
      macroCases,
      samples: macroCases * samplesPerMacro,
      jobs: macroCases * jobsPerMacro,
    };
  }

  function selectTab(name, focus) {
    if (!Object.hasOwn(codeByTab, name)) return;
    activeTab = name;
    document.querySelectorAll("[data-builder-tab]").forEach((button) => {
      const selected = button.dataset.builderTab === name;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
      if (selected && focus) button.focus();
    });
    const activeButton = document.querySelector(`[data-builder-tab="${name}"]`);
    byId("builder-code-panel").setAttribute("aria-labelledby", activeButton.id);
    byId("builder-code").textContent = codeByTab[name];
  }

  function render() {
    const state = readState();
    const campaignState = readCampaignState();
    inputs.name.value = state.name;
    inputs.workers.value = String(state.workers);
    inputs.shardSize.value = String(state.shardSize);
    campaignControls.gpus.value = String(campaignState.gpus);
    campaignControls.days.value = String(campaignState.days);
    campaignControls.utilizationLow.value = String(campaignState.utilizationLow);
    campaignControls.utilizationHigh.value = String(campaignState.utilizationHigh);
    const plan = calculatePlan(state);
    const yaml = buildYaml(state);
    codeByTab = {
      setup: buildSetup(state),
      yaml,
      run: buildRun(state, yaml),
      seawulf: buildSeaWulf(state, yaml),
      campaign: buildCampaignManifest(campaignState),
    };
    byId("builder-macro-cases").textContent = plan.macroCases.toLocaleString();
    byId("builder-samples").textContent = plan.samples.toLocaleString();
    byId("builder-jobs").textContent = plan.jobs.toLocaleString();
    const task = findRow("tasks", state.task);
    byId("builder-task-status").textContent = task.status.replaceAll("_", " ");
    byId("builder-profile-note").textContent = findRow("quality_profiles", state.quality).note;
    addListItems(byId("builder-warnings"), warningMessages(state, plan));
    renderCampaign(campaignState);
    selectTab(activeTab, false);
    updateDeepLink(state, campaignState);
    byId("builder-state").textContent = "Choices saved in this page URL.";
  }

  function reset() {
    Object.values(selectors).forEach((select) => {
      select.value = select.dataset.default || select.options[0].value;
    });
    Object.values(inputs).forEach((input) => {
      input.value = input.defaultValue;
    });
    Object.values(campaignControls).forEach((control) => {
      control.value = control.dataset.default || control.defaultValue;
    });
    activeTab = "setup";
    render();
    byId("builder-state").textContent = "Default choices restored.";
  }

  async function copyCurrentCode() {
    const text = codeByTab[activeTab] || "";
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const helper = document.createElement("textarea");
        helper.value = text;
        helper.setAttribute("readonly", "");
        helper.style.position = "fixed";
        helper.style.opacity = "0";
        document.body.appendChild(helper);
        helper.select();
        const copied = document.execCommand("copy");
        helper.remove();
        if (!copied) throw new Error("copy command unavailable");
      }
      byId("builder-copy-status").textContent = "Copied the current tab.";
    } catch (_error) {
      byId("builder-copy-status").textContent = "Copy was blocked by the browser; select the code and copy it manually.";
    }
  }

  function downloadYaml() {
    const state = readState();
    const blob = new Blob([codeByTab.yaml], { type: "text/yaml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.name}.yaml`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    byId("builder-copy-status").textContent = `Downloaded ${state.name}.yaml.`;
  }

  function bindEvents() {
    form.addEventListener("change", (event) => {
      render();
      if (event.target === selectors.environment && selectors.environment.value === "seawulf") {
        selectTab("seawulf", false);
      }
    });
    [inputs.threshold, inputs.name, inputs.dataRoot, inputs.runRoot, inputs.workers, inputs.shardSize, inputs.group].forEach(
      (input) => input.addEventListener("input", render),
    );
    campaignControls.preset.addEventListener("change", render);
    [
      campaignControls.gpus,
      campaignControls.days,
      campaignControls.utilizationLow,
      campaignControls.utilizationHigh,
    ].forEach((input) => input.addEventListener("input", render));
    byId("builder-reset").addEventListener("click", reset);
    byId("builder-copy").addEventListener("click", copyCurrentCode);
    byId("builder-download").addEventListener("click", downloadYaml);
    const tabs = Array.from(document.querySelectorAll("[data-builder-tab]"));
    tabs.forEach((button, index) => {
      button.addEventListener("click", () => selectTab(button.dataset.builderTab, false));
      button.addEventListener("keydown", (event) => {
        if (!(["ArrowLeft", "ArrowRight"].includes(event.key))) return;
        event.preventDefault();
        const step = event.key === "ArrowRight" ? 1 : -1;
        const next = tabs[(index + step + tabs.length) % tabs.length];
        selectTab(next.dataset.builderTab, true);
      });
    });
  }

  fetch("../assets/benchmark-builder-options.json", { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((payload) => {
      options = payload;
      Object.values(selectors).forEach(populateSelect);
      populateSelect(campaignControls.preset);
      applyDeepLink();
      renderQualityTable();
      bindEvents();
      render();
    })
    .catch((error) => {
      byId("builder-code").textContent = "The benchmark contract could not be loaded. Refresh the page or use the static server guide.";
      byId("builder-state").textContent = `Builder unavailable: ${error.message}`;
    });
})();
