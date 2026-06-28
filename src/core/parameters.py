# parameters.py

"""
All constants and configuration settings for the ELICIT (Emergent LLM Institutions for Climate and International Treaties) simulation.
"""

# --- Simulation Settings ---
NUM_AGENTS = 7           # Total number of participants
NUM_ROUNDS = 30          # Total number of rounds per simulation
SEED = 42                # Random seed for reproducibility
SCENARIO = "abstract"    # Formatting scenario for prompts (abstract, ldf, tax)
AGENT_TYPE = "LLM"       # Control baseline (LLM, Random, Greedy)
BATCH_NAME = "Control"   # Current experiment batch name
MIXED_AGENT_COUNTS = None # Optional composition dict, e.g. {"LLM": 5, "Random": 1, "Greedy": 1}
CURRENT_RUN = 0           # Progress counter for multi-run experiment scripts
TOTAL_RUNS = 0            # Total planned runs for progress reporting

# --- LLM Configuration (Local Ollama) ---
LLM_MODEL = "llama3.1:8b"
LLM_BASE_URL = "http://localhost:11434/v1"
OLLAMA_REQUEST_TIMEOUT_SECONDS = 3000.0
LLM_MAX_CONCURRENCY = 2
# Ollama runtime options forwarded on every request (native + OpenAI-compatible API).
# num_gpu: model layers offloaded to GPU (Ollama option name is num_gpu).
# num_ctx: KV-cache reservation — set close to your longest prompt, not higher than needed.
OLLAMA_NUM_GPU = -1
OLLAMA_NUM_CTX = 4096
# Parallel slots in the Ollama server process — set the same value when starting `ollama serve`.
OLLAMA_NUM_PARALLEL = 1

# --- Initial Endowments ---
INITIAL_TOKENS = 1000    # Starting tokens per agent
ENDOWMENT_STAGE_1 = 20   # Tokens per agent per round (Stage 1 contribution)
ENDOWMENT_STAGE_2 = 20   # Tokens per agent per round (Stage 2 sanctions)
STAGE_2_WEALTH_FRACTION = 0.05  # Climate/LDF: stage-2 sanction budget = max(ENDOWMENT_STAGE_2, wealth * this)

# --- Public Goods Game ---
PUBLIC_GOOD_MULTIPLIER = 1.6   # Multiplication factor for group contributions (MCPR = this / group_size)
MIN_CONTRIBUTION = 0
MAX_CONTRIBUTION = ENDOWMENT_STAGE_1

# --- Punishment & Reward Settings (SI only) ---
MAX_PUNISHMENT_TOKENS = 20     # Max tokens an agent can assign in Stage 2
PUNISHMENT_EFFECT = 3          # Negative token reduces target payoff by 3
PUNISHMENT_COST = 1            # Each negative token costs 1 from sender
REWARD_EFFECT = 1              # Positive token increases target payoff by 1
REWARD_COST = 1                # Each positive token costs 1 from sender

# Keep disabled unless explicitly enabled in experiments.
RULE_OF_LAW_ENABLED = False

# --- Information Settings ---
ANONYMITY = False              # Agents don't see others anonymously (identity tracking)
DISPLAY_PAST_ACTIONS = 1       # Controls how many rounds of peer data history are kept (T-1 window); agents use belief_state for long-term memory

# --- Belief Tracking (Working Memory / Scratchpad) ---
BELIEF_TRACKING_ENABLED = True # Enable structured belief-state updates after each round

# --- Phase 2: Cognitive Modules ---
TOM_ENABLED = True             # Enable Theory of Mind audits after each round
TOM_VERBOSE = True             # Log each agent's published trust scores to the terminal
DEMOCRACY_ENABLED = True       # Enable democratic rule-changing every N rounds
DEMOCRACY_INTERVAL = 5        # Rounds between constitutional votes

GOSSIP_ENABLED = True          # Distribute negative ToM audits as social pressure
GOSSIP_TRIGGER_SCORE = 7.0     # Only share gossip for trust scores <= this value
MAX_GOSSIP_ITEMS = 5           # Prevent prompt bloat

# --- Phase 3: Oracle Settings ---
ORACLE_PUNISHMENT_WEIGHT = 1.5
ORACLE_REWARD_WEIGHT = 1.2
ORACLE_ENDOWMENT_SCALING = 50.0
ORACLE_MAX_TOKENS_WEIGHT = 60.0

# --- Phase 4: Subsidy & Curiosity ---
SUBSIDY_ENABLED = True       # Toggle subsidy redistribution
SUBSIDY_FRACTION = 0.2       # 20% of punishment costs pooled for subsidy
SUBSIDY_TOP_N = 2            # Number of top contributors who receive subsidy

CURIOSITY_ENABLED = False      # Toggle LLM-driven curiosity
CURIOSITY_BONUS_PROMPT = False # If True, injects novelty suggestions into prompts

# --- Output Settings ---
SAVE_RESULTS = True            # Save results to file after simulation
VERBOSE = True                 # Print detailed round-by-round logs

# --- Phase 5: Heterogeneous Climate Economy ---
# Counts should sum to NUM_AGENTS for deterministic profile assignment.
AGENT_GROUP_COUNTS = {
	"developed": 3,
	"developing": 4,
}

# LDF-specific country composition and initial endowment profile.
# Major developed-country values are grounded in commonly cited initial LDF pledges.
LDF_AGENT_GROUP_COUNTS = {
	"developed": 12,
	"developing": 14,
}

# Developed-country initial endowments (million USD) — using nominal GDP (no scaling):
# United States, Germany, Japan, United Kingdom, France, Italy, Canada, Australia,
# Netherlands, Switzerland, Ireland, Sweden (values in million USD)
LDF_DEVELOPED_INITIAL_ENDOWMENTS = [
	28750000.0, 4690000.0, 4030000.0, 3690000.0, 3160000.0, 2380000.0,
	2240000.0, 1760000.0, 1210000.0, 936560.0, 609160.0, 603720.0
]

# Developing-country initial endowments (million USD) — using nominal GDP (no scaling):
# South Africa, Egypt, Ethiopia, Saudi Arabia, UAE, Pakistan, Brazil, Colombia,
# Ecuador, Maldives, Antigua and Barbuda, Nepal, Senegal, Armenia
LDF_DEVELOPING_INITIAL_ENDOWMENTS = [
	377700.0, 380060.0, 163700.0, 1110000.0, 532700.0, 407790.0,
	2330000.0, 363500.0, 118800.0, 6600.0, 2120.0, 44180.0, 33700.0, 25410.0
]

# Backward-compatible fallback if a flat developing value is preferred elsewhere.
LDF_DEVELOPING_INITIAL_WEALTH = 5.0

DEVELOPED_INITIAL_WEALTH = 1300.0
DEVELOPING_INITIAL_WEALTH = 700.0

DEVELOPED_VULNERABILITY = 0.30
DEVELOPING_VULNERABILITY = 1.50

DEVELOPED_HISTORICAL_EMISSIONS = 1.50
DEVELOPING_HISTORICAL_EMISSIONS = 0.30

DEVELOPED_CONTRIBUTION_CAPACITY = 1.00
DEVELOPING_CONTRIBUTION_CAPACITY = 0.10

# --- Phase 5a: Climate Shock Dynamics ---
CLIMATE_SHOCK_ENABLED = False
CLIMATE_SHOCK_BASE_PROB = 0.20
CLIMATE_SHOCK_SEVERITY_MIN = 0.05
CLIMATE_SHOCK_SEVERITY_MAX = 0.25
CLIMATE_DAMAGE_BASE = 150000.0

# Optional deterministic shock mode for controlled testing of shock impacts and agent responses.
# When True, the environment will ignore the probabilistic shock roll and
# instead trigger shocks according to `CLIMATE_SHOCK_SCHEDULE`.
# Schedule entries are (round_number, severity) tuples where severity is a
# scalar in the same range as the stochastic severity (e.g. 0.10 = 10%).
CLIMATE_SHOCK_DETERMINISTIC = True
# Schedule: shocks at round 5 (10%) and round 10 (20%).
CLIMATE_SHOCK_SCHEDULE = [
	(5, 0.10),
	(10, 0.20),
]

# --- Phase 5b: Loss & Damage Fund (LDF) ---
LDF_ENABLED = False

# Contribution policy (scalar policy; can be modified by democracy)
LDF_COLLECT_EVERY_ROUND = True
LDF_REPLENISHMENT_INTERVAL = 5

# Payout policy (scalar policy; can be modified by democracy)
LDF_PAYOUT_DAMAGE_WEIGHT = 1.0
LDF_MAX_COVERAGE = 0.90
LDF_EQUITY_WEIGHT = 0.0
