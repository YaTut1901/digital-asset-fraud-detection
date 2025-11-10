# 🏦 Static Pool Data
| id | Feature | Description | Derivation | Type | Source |
|----|----------|--------------|-------------|------|--------|
| 1 | pool_address | Address/Id of the liquidity pool (if any) | Pool address | String | Graph |
| 2 | chain_name | Blockchain (e.g. Ethereum, Solana) | Chain name | String | CSV |
| 3 | chain_id | Numeric or symbolic chain identifier | Chain id | Integer/String | CSV |
| 4 | token0_address | Address of first token in pair | Token address | String | CSV, Graph |
| 5 | token1_address | Address of second token in pair | Token address | String | CSV, Graph |
| 6 | token0_pool_amount | Amount of pools on DEX with first token | Count fetched pools | Integer | Graph |
| 7 | token1_pool_amount | Amount of pools on DEX with second token | Count fetched pools | Integer | Graph |
| 8 | token0_age_minutes | Time from first token contract creation to DEX listing | Get pool creation timestamp from The Graph, get token contract creation timestamp from Etherscan getcontractcreation (includes timestamp), calculate difference in minutes | Integer (minutes) | Etherscan, Graph |
| 9 | token1_age_minutes | Time from second token contract creation to DEX listing | Get pool creation timestamp from The Graph, get token contract creation timestamp from Etherscan getcontractcreation (includes timestamp), calculate difference in minutes | Integer (minutes) | Etherscan, Graph |
| 10 | first_pool_activity_time | Time from DEX listing to first pool LP action | Query Dune with single aggregated SQL query of Mint action, filtered by pool addresses | Integer (minutes) | Dune, Flipside |
| 11 | dex_name | Name of DEX where pool is created | DEX name | String | CSV |
| 12 | token0_max_supply | Maximum supply defined in first token contract | Use batched eth_call of totalSupply() at your target block. The fastest way is a Multicall at that block; fall back to direct calls if the multicall contract didn't exist yet at that block. | Float | RPC |
| 13 | token1_max_supply | Maximum supply defined in second token contract | Use batched eth_call of totalSupply() at your target block. The fastest way is a Multicall at that block; fall back to direct calls if the multicall contract didn't exist yet at that block. | Float | RPC |
| 14 | token0_minting_enabled | Can the first token be minted after deployment? | Check if there is a value in default proxy slot, if yes then fetch bytecode from RPC for this value, if no fetch bytecode from RPC for original address, execute bytecode with mythril, look in execution graph for pattern that corresponds to mint function | Boolean | RPC, Mythril |
| 15 | token1_minting_enabled | Can the second token be minted after deployment? | Check if there is a value in default proxy slot, if yes then fetch bytecode from RPC for this value, if no fetch bytecode from RPC for original address, execute bytecode with mythril, look in execution graph for pattern that corresponds to mint function | Boolean | RPC, Mythril |
| 16 | token0_verified_contract | Is the first token contract verified on-chain? | Call Etherscan getsourcecode endpoint for proxy and for implementation | Boolean | Etherscan |
| 17 | token1_verified_contract | Is the second token contract verified on-chain? | Call Etherscan getsourcecode endpoint for proxy and for implementation | Boolean | Etherscan |
| 18 | token0_ownership_renounced | Is ownership of the first token contract renounced? | Use batched eth_call of owner() function. The fastest way is a Multicall at that block; fall back to direct calls if the multicall contract didn't exist yet at that block. | Boolean | RPC |
| 19 | token1_ownership_renounced | Is ownership of the second token contract renounced? | Use batched eth_call of owner() function. The fastest way is a Multicall at that block; fall back to direct calls if the multicall contract didn't exist yet at that block. | Boolean | RPC |
| 20 | token0_proxy | Whether the first token uses a proxy | Call Etherscan getsourcecode endpoint, check if Proxy field equals '1' | Boolean | Etherscan |
| 21 | token0_pausable | If the first token has pausable functionality | Check if there is a value in default proxy slot, if yes then fetch bytecode from RPC for this value, if no fetch bytecode from RPC for original address, execute bytecode with mythril, look in execution graph for pattern that corresponds to pause function (OZ Pausable) | Boolean | RPC, Mythril |
| 22 | token0_burning_enabled | Can the first token be burned after deployment? | Check if there is a value in default proxy slot, if yes then fetch bytecode from RPC for this value, if no fetch bytecode from RPC for original address, execute bytecode with mythril, look in execution graph for pattern that corresponds to burn function | Boolean | RPC, Mythril |
| 23 | token1_proxy | Whether the second token uses a proxy | Call Etherscan getsourcecode endpoint, check if Proxy field equals '1' | Boolean | Etherscan |
| 24 | token1_pausable | If the second token has pausable functionality | Check if there is a value in default proxy slot, if yes then fetch bytecode from RPC for this value, if no fetch bytecode from RPC for original address, execute bytecode with mythril, look in execution graph for pattern that corresponds to pause function (OZ Pausable) | Boolean | RPC, Mythril |
| 25 | token1_burning_enabled | Can the second token be burned after deployment? | Check if there is a value in default proxy slot, if yes then fetch bytecode from RPC for this value, if no fetch bytecode from RPC for original address, execute bytecode with mythril, look in execution graph for pattern that corresponds to burn function | Boolean | RPC, Mythril |
| 26 | token0_owner_is_contract | Is the first token owner address a contract? | Call getCode on token owner address; if it returns non-empty bytecode, it's a contract | Boolean | RPC |
| 27 | token1_owner_is_contract | Is the second token owner address a contract? | Call getCode on token owner address; if it returns non-empty bytecode, it's a contract | Boolean | RPC |
| 28 | pool_deployer_is_contract | Is the pool deployer address a contract? | Call getCode on deployer address; if it returns non-empty bytecode, it's a contract | Boolean | RPC |
| 29 | token0_owner_tx_count | Amount of transactions from the owner address | If owner address is the contract (getCode returns not 0) then call Etherscan account txlist API to count transactions to the contract, otherwise call eth_getTransactionCount | Integer | Etherscan, RPC |
| 30 | token1_owner_tx_count | Amount of transactions from the owner address | If owner address is the contract (getCode returns not 0) then call Etherscan account txlist API to count transactions to the contract, otherwise call eth_getTransactionCount | Integer | Etherscan, RPC |
| 31 | pool_deployer_tx_count | Amount of transactions from the pool deployer address | If deployer address is the contract (getCode returns not 0) then call Etherscan account txlist API to count transactions to the contract, otherwise call eth_getTransactionCount | Integer | Etherscan, RPC |
| 32 | token0_owner_age | Time passed from first token owner address's first tx | If owner address is the contract (getCode returns not 0) then call Etherscan getcontractcreation, otherwise call Etherscan account txlist API with offset=1 and sort=asc to get first transaction directly | Integer (minutes) | Etherscan |
| 33 | token1_owner_age | Time passed from second token owner address's first tx | If owner address is the contract (getCode returns not 0) then call Etherscan getcontractcreation, otherwise call Etherscan account txlist API with offset=1 and sort=asc to get first transaction directly | Integer (minutes) | Etherscan |
| 34 | pool_deployer_age | Time passed from pool deployer address's first tx | If deployer address is the contract (getCode returns not 0) then call Etherscan getcontractcreation, otherwise call Etherscan account txlist API with offset=1 and sort=asc to get first transaction directly | Integer (minutes) | Etherscan |
| 35 | token0_owner_gas_burnt | Amount of gas burnt by first token owner address | Call Etherscan account txlist API to get all transactions, sum gasUsed for total gas units consumed | Integer | Etherscan |
| 36 | token1_owner_gas_burnt | Amount of gas burnt by second token owner address | Call Etherscan account txlist API to get all transactions, sum gasUsed for total gas units consumed | Integer | Etherscan |
| 37 | pool_deployer_gas_burnt | Amount of gas burnt by pool deployer address | Call Etherscan account txlist API to get all transactions, sum gasUsed for total gas units consumed | Integer | Etherscan |
| 38 | token0_top_10_holders_percent | Supply of first token owned by top 10 holders in percent | Calculate percent of summed top 10 holders supply at the moment of pool creation | Float (%) | Bitquery |
| 39 | token1_top_10_holders_percent | Supply of second token owned by top 10 holders in percent | Calculate percent of summed top 10 holders supply at the moment of pool creation | Float (%) | Bitquery |
| 40 | token0_num_holder_launch | Number of first token holders at the moment of pool creation | Count holders at the moment pool is created | Integer | Bitquery |
| 41 | token1_num_holder_launch | Number of second token holders at the moment of pool creation | Count holders at the moment pool is created | Integer | Bitquery |
| 42 | token0_liquidity_depth | Percent of first token total supply locked in pool | Call token contract balanceOf(poolAddress) at pool creation block, divide by token totalSupply() at same block, multiply by 100 for percentage | Float (%) | RPC |
| 43 | token1_liquidity_depth | Percent of second token total supply locked in pool | Call token contract balanceOf(poolAddress) at pool creation block, divide by token totalSupply() at same block, multiply by 100 for percentage | Float (%) | RPC |
| 44 | label | Whether pool contains fraud asset | True if one token is labeled as scam, false if both are trustworthy (labeled or in list on https://tokens.uniswap.org/), undefined otherwise | Boolean | CSV |

## Pseudocode: Single Pool Metrics (batched, cached; one call per metric)

```python
# Caches used across metrics (memoized per address/block where applicable)
caches = {
    'etherscan_source': {},          # addr -> source json
    'etherscan_creation': {},        # addr -> {'timestamp', 'blockNumber', 'creator'}
    'rpc_code': {},                  # addr -> bytecode
    'rpc_storage': {},               # (addr, slot, block) -> value
    'multicall': {},                 # (addr, sig, args, block) -> return
    'txlist': {},                    # addr -> [txs]
    'mythril': {},                   # addr_or_impl -> analysis
    'bitquery_holders': {},          # (token, block) -> holders list
    'graph_pool_ctx': {},            # pool -> {token0, token1, createdAt, createdAtBlock, deployer}
}

def calculate_single_pool_metrics(pool_address):
    metrics = {}

    # Constants per your spec
    metrics['chain_name'] = derive_chain_name_const("ethereum")                # id 2
    metrics['chain_id'] = derive_chain_id_const(1)                             # id 3
    metrics['dex_name'] = derive_dex_name_const("Uniswap")                     # id 11
    metrics['pool_address'] = derive_pool_address(pool_address)                # id 1

    # Pool context (Graph) - one fetch reused by many metrics
    pool_ctx = graph_get_pool_context(pool_address, caches)  # {token0, token1, createdAt, createdAtBlock, deployer}
    token0 = pool_ctx['token0']
    token1 = pool_ctx['token1']
    pool_created_ts = pool_ctx['createdAt']
    pool_created_block = pool_ctx['createdAtBlock']
    pool_deployer = pool_ctx['deployer']

    # Token addresses (Graph)
    metrics['token0_address'] = derive_token0_address(token0)                  # id 4
    metrics['token1_address'] = derive_token1_address(token1)                  # id 5

    # Count all existing pools for each token (Graph)
    metrics['token0_pool_amount'] = derive_token_pool_amount(token0, caches)   # id 6
    metrics['token1_pool_amount'] = derive_token_pool_amount(token1, caches)   # id 7

    # Contract creation timestamps (Etherscan) for token ages
    t0_creation = etherscan_get_contract_creation_cached(token0, caches)
    t1_creation = etherscan_get_contract_creation_cached(token1, caches)
    metrics['token0_age_minutes'] = derive_token_age_minutes(pool_created_ts, t0_creation['timestamp'])  # id 8
    metrics['token1_age_minutes'] = derive_token_age_minutes(pool_created_ts, t1_creation['timestamp'])  # id 9

    # First pool activity time (Dune) - earliest Mint after pool creation
    first_mint_ts = dune_get_first_mint_timestamp(pool_address)
    metrics['first_pool_activity_time'] = derive_first_pool_activity_minutes(pool_created_ts, first_mint_ts)  # id 10

    # Etherscan source (verification + proxy) - fetched once per token, reused
    t0_source = etherscan_get_source_code_cached(token0, caches)
    t1_source = etherscan_get_source_code_cached(token1, caches)
    metrics['token0_verified_contract'] = derive_verified_from_source(t0_source)   # id 16
    metrics['token1_verified_contract'] = derive_verified_from_source(t1_source)   # id 17
    metrics['token0_proxy'] = derive_proxy_flag_from_source(t0_source)             # id 20
    metrics['token1_proxy'] = derive_proxy_flag_from_source(t1_source)             # id 23

    # Multicall batch at pool creation block: totalSupply, owner, balanceOf(pool)
    mc = multicall_batch_at_block([
        (token0, 'totalSupply()'), (token1, 'totalSupply()'),
        (token0, 'owner()'), (token1, 'owner()'),
        (token0, 'balanceOf(address)', pool_address),
        (token1, 'balanceOf(address)', pool_address),
    ], block=pool_created_block, caches=caches)

    t0_total = derive_token_total_supply_from_mc(token0, mc)                   # id 12
    t1_total = derive_token_total_supply_from_mc(token1, mc)                   # id 13
    t0_owner = derive_token_owner_from_mc(token0, mc)                          # used by ids 18,26,29,32,35
    t1_owner = derive_token_owner_from_mc(token1, mc)                          # used by ids 19,27,30,33,36
    t0_pool_bal = derive_token_balance_in_pool_from_mc(token0, pool_address, mc)  # used by id 42
    t1_pool_bal = derive_token_balance_in_pool_from_mc(token1, pool_address, mc)  # used by id 43

    metrics['token0_max_supply'] = t0_total                                    # id 12
    metrics['token1_max_supply'] = t1_total                                    # id 13
    metrics['token0_ownership_renounced'] = derive_ownership_renounced(t0_owner)   # id 18
    metrics['token1_ownership_renounced'] = derive_ownership_renounced(t1_owner)   # id 19

    # Is owner/deployer a contract? (RPC getCode)
    metrics['token0_owner_is_contract'] = derive_is_contract(t0_owner, caches)     # id 26
    metrics['token1_owner_is_contract'] = derive_is_contract(t1_owner, caches)     # id 27
    metrics['pool_deployer_is_contract'] = derive_is_contract(pool_deployer, caches)  # id 28

    # Proxy implementation address for Mythril analysis (if proxy)
    t0_impl = get_impl_for_analysis(token0, metrics['token0_proxy'], pool_created_block, caches)
    t1_impl = get_impl_for_analysis(token1, metrics['token1_proxy'], pool_created_block, caches)

    # Mythril analysis (heavy) on implementation bytecode
    t0_myth = mythril_analyze_cached(t0_impl, caches)
    t1_myth = mythril_analyze_cached(t1_impl, caches)
    metrics['token0_minting_enabled'] = derive_minting_enabled(t0_myth)        # id 14
    metrics['token1_minting_enabled'] = derive_minting_enabled(t1_myth)        # id 15
    metrics['token0_pausable'] = derive_pausable_enabled(t0_myth)              # id 21
    metrics['token1_pausable'] = derive_pausable_enabled(t1_myth)              # id 24
    metrics['token0_burning_enabled'] = derive_burning_enabled(t0_myth)        # id 22
    metrics['token1_burning_enabled'] = derive_burning_enabled(t1_myth)        # id 25

    # TX counts (Etherscan/RPC) and age/gas per owner/deployer
    for prefix, address, is_contract in [
        ('token0_owner', t0_owner, metrics['token0_owner_is_contract']),
        ('token1_owner', t1_owner, metrics['token1_owner_is_contract']),
        ('pool_deployer', pool_deployer, metrics['pool_deployer_is_contract']),
    ]:
        # Age minutes
        metrics[f'{prefix}_age'] = derive_address_age_minutes(address, is_contract, pool_created_ts, caches)  # ids 32,33,34
        # TX count
        metrics[f'{prefix}_tx_count'] = derive_tx_count(address, is_contract, caches)                         # ids 29,30,31
        # Total gas units consumed (sum gasUsed)
        metrics[f'{prefix}_gas_burnt'] = derive_gas_burnt_units(address, caches)                              # ids 35,36,37

    # Liquidity depth at creation (RPC values already loaded via multicall)
    metrics['token0_liquidity_depth'] = derive_liquidity_depth_percent(t0_pool_bal, t0_total)  # id 42
    metrics['token1_liquidity_depth'] = derive_liquidity_depth_percent(t1_pool_bal, t1_total)  # id 43

    # Holders at creation (Bitquery), also top-10 %
    t0_holders = bitquery_get_holders_at_block_cached(token0, pool_created_block, caches)
    t1_holders = bitquery_get_holders_at_block_cached(token1, pool_created_block, caches)
    metrics['token0_top_10_holders_percent'] = derive_top10_percent(t0_holders, t0_total)      # id 38
    metrics['token1_top_10_holders_percent'] = derive_top10_percent(t1_holders, t1_total)      # id 39
    metrics['token0_num_holder_launch'] = derive_num_holders(t0_holders)                       # id 40
    metrics['token1_num_holder_launch'] = derive_num_holders(t1_holders)                       # id 41

    # Label (CSV)
    metrics['label'] = derive_label_from_csv(token0, token1)                                   # id 44

    return metrics

# ---- Helper notes (each derive_* maps 1:1 to a metric) ----
# - graph_get_pool_context: one fetch reused (token addresses, createdAt, createdAtBlock, deployer)
# - etherscan_get_*_cached: memoized; parallelize per address list
# - multicall_batch_at_block: one call returning all token state at pool creation block
# - get_impl_for_analysis: if proxy True -> read EIP-1967 slot via rpc_get_storage_at; else token addr
# - mythril_analyze_cached: memoized per implementation address
# - derive_tx_count: contract -> use Etherscan txlist (to=address); EOA -> eth_getTransactionCount
# - derive_address_age_minutes: contract -> creation timestamp; EOA -> first tx (txlist offset=1, sort=asc)
# - derive_gas_burnt_units: sum gasUsed from Etherscan txlist for the address
```

## Pseudocode: Multi-Token → Multi-Pool Pipeline (efficient, batched, cached)

```python
def calculate_metrics_for_token_list(csv_path):
    # Read tokens from categorized_tokens.csv
    tokens = load_categorized_tokens(csv_path)  # returns [token_address, name, symbol, label]

    # Constants 
    chain_name = "ethereum"
    chain_id = 1
    dex_name = "Uniswap"

    # ---- Step A: Fetch all Uniswap pools for each token (batch Graph queries) ----
    # Uniswap v2: pairs; Uniswap v3: pools
    v2_pairs_by_token = graph_batch_list_v2_pairs_for_tokens(tokens)   # {token -> [pair_addr]}
    v3_pools_by_token = graph_batch_list_v3_pools_for_tokens(tokens)   # {token -> [pool_addr]}

    # Build the global set of pools for processing
    all_pools = list()
    for t in tokens:
        for p in v2_pairs_by_token.get(t, []): all_pools.add(p)
        for p in v3_pools_by_token.get(t, []): all_pools.add(p)

    # ---- Step B: Prefetch pool contexts (Graph) for all pools (token0, token1, createdAt, block, deployer ----
    graph_prefetch_pool_contexts(all_pools, caches)  # fills caches['graph_pool_ctx']

    # Collect unique token addresses seen across all pools (to maximize reuse)
    tokens_in_scope = collect_unique_tokens_from_pool_contexts(all_pools, caches)  # set([token0, token1, ...])

    # ---- Step C: Batch prefetch Etherscan data for all tokens in scope ----
    etherscan_prefetch_source(tokens_in_scope, caches)     # verification + proxy flags
    etherscan_prefetch_creation(tokens_in_scope, caches)   # creation timestamps

    # ---- Step D: Dune prefetch for first-mint timestamps across all pools ----
    dune_prefetch_first_mint_timestamps(all_pools)         # pool_addr -> ts

    # ---- Step E: Multicall prefetch at pool-creation block (grouped in chunks) ----
    # For each pool, we need: token0.totalSupply(), token1.totalSupply(),
    #                         token0.owner(), token1.owner(),
    #                         token0.balanceOf(pool), token1.balanceOf(pool)
    mc_requests = build_multicall_requests_for_pools(all_pools, caches)
    multicall_prefetch_token_state_at_blocks(mc_requests, caches)  # fills caches['multicall']

    # ---- Step F: Compute metrics per pool (parallel with bounded concurrency) ----
    results = []
    for pool in parallel_iter(all_pools, max_concurrency=K()):
        # Reuse shared caches; single-pool function maps 1:1 derive_* calls to metrics
        m = calculate_single_pool_metrics(pool)
        # Override constants as per spec to avoid drift
        m['chain_name'] = chain_name
        m['chain_id'] = chain_id
        m['dex_name'] = dex_name
        results.append(m)

    return results

# Notes:
# - graph_batch_list_v2_pairs_for_tokens and graph_batch_list_v3_pools_for_tokens should use token-in filter batching.
# - graph_prefetch_pool_contexts hydrates token0, token1, createdAt, createdAtBlock, deployer for all pools.
# - etherscan_prefetch_* runs in chunks with rate limiting; fills caches to be reused by per-pool derivations.
# - multicall_prefetch_token_state_at_blocks groups requests by block and token to minimize calls.
# - parallel_iter runs pool computations with a small concurrency limit and shared caches, eliminating duplicates.
```