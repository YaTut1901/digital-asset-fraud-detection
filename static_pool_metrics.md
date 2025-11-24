# 🏦 Static Pool Data
| id | Feature | Description | Derivation | Type | Source | Pseudocode Added | Assigned |
|----|----------|--------------|-------------|------|--------|-------------------|----------|
| 1  | pool_address               | Address/Id of the liquidity pool (if any)                                                         | Pool address                                                                         | String           | Graph                  | ✅ | s.hofryk|
| 2  | dex_name                   | Name of DEX where pool is created                                                                 | DEX name                                                                             | String           | CSV                    | ✅ | |
| 3  | chain_name                 | Blockchain (e.g. Ethereum, Solana)                                                               | Chain name                                                                           | String           | CSV                    | ✅ | |
| 4  | chain_id                   | Numeric or symbolic chain identifier                                                              | Chain id                                                                             | Integer/String   | CSV                    | ✅ | |
| 5  | first_pool_activity_time   | Time from DEX listing to first pool LP action                                                     | Query Dune with single aggregated SQL query of Mint action, filtered by pool addresses| Integer (minutes)| Dune, Flipside         | ✅ | |
| 6  | token0_address             | Address of first token in pair                                                                    | Token address                                                                        | String           | CSV, Graph             | ✅ | s.hofryk|
| 7  | token1_address             | Address of second token in pair                                                                   | Token address                                                                        | String           | CSV, Graph             | ✅ | s.hofryk|
| 8  | token0_name                | Name of the first token in pair                                                                   | Fetch via Graph, RPC, or CSV (preferably from token contract's name() or metadata)   | String           | CSV, Graph, RPC        | ✅ | s.hofryk |
| 9  | token1_name                | Name of the second token in pair                                                                  | Fetch via Graph, RPC, or CSV (preferably from token contract's name() or metadata)   | String           | CSV, Graph, RPC        | ✅ | s.hofryk |
| 10 | token0_symbol              | Symbol of the first token in pair                                                                 | Fetch via Graph, RPC, or CSV (preferably from token contract's symbol() or metadata) | String           | CSV, Graph, RPC        | ✅ | s.hofryk|
| 11 | token1_symbol              | Symbol of the second token in pair                                                                | Fetch via Graph, RPC, or CSV (preferably from token contract's symbol() or metadata) | String           | CSV, Graph, RPC        | ✅ | s.hofryk|
| 12 | token0_pool_amount         | Amount of pools on DEX with first token                                                           | Count fetched pools                                                                  | Integer          | Graph                  | ✅ | s.hofryk|
| 13 | token1_pool_amount         | Amount of pools on DEX with second token                                                          | Count fetched pools                                                                  | Integer          | Graph                  | ✅ | s.hofryk|
| 14 | token0_total_supply        | Total supply defined in first token contract                                                     | Use batched eth_call of totalSupply() at your target block. Multicall/fallback logic | Float            | RPC                    | ✅ | | Olena|
| 15 | token1_total_supply        | Total supply defined in second token contract                                                    | Use batched eth_call of totalSupply() at your target block. Multicall/fallback logic | Float            | RPC                    | ✅ | | Olena|
| 16 | token0_age_minutes         | Time from first token contract creation to DEX listing                                            | Get pool creation timestamp from The Graph, get token contract creation timestamp from Etherscan, calculate age in minutes  | Integer (minutes)| Etherscan, Graph       | ✅ | s.hofryk|
| 17 | token1_age_minutes         | Time from second token contract creation to DEX listing                                           | Get pool creation timestamp from The Graph, get token contract creation timestamp from Etherscan, calculate age in minutes | Integer (minutes)| Etherscan, Graph       | ✅ | s.hofryk|
| 18 | token0_verified_contract   | Is the first token contract verified on-chain?                                                    | Call Etherscan getsourcecode endpoint for proxy and for implementation               | Boolean          | Etherscan              | ✅ | |
| 19 | token1_verified_contract   | Is the second token contract verified on-chain?                                                   | Call Etherscan getsourcecode endpoint for proxy and for implementation               | Boolean          | Etherscan              | ✅ | |
| 20 | token0_ownership_renounced | Is ownership of the first token contract renounced?                                               | Use eth_call of owner(), via multicall/direct call                                   | Boolean          | RPC                    | ✅ | |
| 21 | token1_ownership_renounced | Is ownership of the second token contract renounced?                                              | Use eth_call of owner(), via multicall/direct call                                   | Boolean          | RPC                    | ✅ | |
| 22 | token0_proxy               | Whether the first token uses a proxy                                                              | Etherscan getsourcecode, Proxy field check                                           | Boolean          | Etherscan              | ✅ | |
| 23 | token1_proxy               | Whether the second token uses a proxy                                                             | Etherscan getsourcecode, Proxy field check                                           | Boolean          | Etherscan              | ✅ | |
| 24 | token0_owner_is_contract   | Is the first token owner address a contract?                                                      | Call getCode on token owner address                                                  | Boolean          | RPC                    | ✅ | |
| 25 | token1_owner_is_contract   | Is the second token owner address a contract?                                                     | Call getCode on token owner address                                                  | Boolean          | RPC                    | ✅ | |
| 26 | pool_deployer_is_contract  | Is the pool deployer address a contract?                                                          | Call getCode on deployer address                                                     | Boolean          | RPC                    | ✅ | |
| 27 | token0_owner_tx_count      | Amount of transactions from the owner address                                                     | Dune aggregated tx count; if contract then incoming, else outgoing                   | Integer          | Dune                   | ✅ |Bohdan |
| 28 | token1_owner_tx_count      | Amount of transactions from the owner address                                                     | Dune aggregated tx count; if contract then incoming, else outgoing                   | Integer          | Dune                   | ✅ |Bohdan |
| 29 | pool_deployer_tx_count     | Amount of transactions from the pool deployer address                                             | Dune aggregated tx count; if contract then incoming, else outgoing                   | Integer          | Dune                   | ✅ |Bohdan |
| 30 | token0_owner_age           | Time passed from first token owner address's first tx                                             | Dune earliest tx timestamp; compute minutes                                          | Integer (minutes)| Dune                   | ✅ |Bohdan |
| 31 | token1_owner_age           | Time passed from second token owner address's first tx                                            | Dune earliest tx timestamp; compute minutes                                          | Integer (minutes)| Dune                   | ✅ |Bohdan |
| 32 | pool_deployer_age          | Time passed from pool deployer address's first tx                                                 | Dune earliest tx timestamp; compute minutes                                          | Integer (minutes)| Dune                   | ✅ |Bohdan |
| 33 | token0_owner_gas_burnt     | Amount of gas burnt by first token owner address                                                  | Dune sum of gasUsed (per direction rule above)                                       | Integer          | Dune                   | ✅ |Bohdan |
| 34 | token1_owner_gas_burnt     | Amount of gas burnt by second token owner address                                                 | Dune sum of gasUsed (per direction rule above)                                       | Integer          | Dune                   | ✅ |Bohdan |
| 35 | pool_deployer_gas_burnt    | Amount of gas burnt by pool deployer address                                                      | Dune sum of gasUsed (per direction rule above)                                       | Integer          | Dune                   | ✅ |Bohdan |
| 36 | token0_owner_bytes_deployed   | Total bytes of code deployed by the first token owner address                                   | Sum code size for all contracts created by the owner                                   | Integer (bytes)     | Dune, RPC               | ✅ | |
| 37 | token1_owner_bytes_deployed   | Total bytes of code deployed by the second token owner address                                  | Sum code size for all contracts created by the owner                                   | Integer (bytes)     | Dune, RPC               | ✅ | |
| 38 | pool_deployer_bytes_deployed  | Total bytes of code deployed by the pool deployer address                                       | Sum code size for all contracts created by the deployer                                | Integer (bytes)     | Dune, RPC               | ✅ | |
| 39 | token0_owner_smart_contracts_interacted | Number of unique smart contracts the first token owner has interacted with                | Dune SQL: count distinct contract addresses in 'to_address' for owner, filter out EOAs        | Integer             | Dune, RPC                | ✅ | |
| 40 | token1_owner_smart_contracts_interacted | Number of unique smart contracts the second token owner has interacted with               | Dune SQL: count distinct contract addresses in 'to_address' for owner, filter out EOAs        | Integer             | Dune, RPC                | ✅ | |
| 41 | pool_deployer_smart_contracts_interacted | Number of unique smart contracts the pool deployer has interacted with                    | Dune SQL: count distinct contract addresses in 'to_address' for deployer, filter out EOAs     | Integer             | Dune, RPC                | ✅ | |
| 42 | token0_top_10_holders_percent | Supply of first token owned by top 10 holders in percent                                       | Calculate percent of summed top 10 holders supply at pool creation                   | Float (%)        | Bitquery               | ✅ | |
| 43 | token1_top_10_holders_percent | Supply of second token owned by top 10 holders in percent                                      | Calculate percent of summed top 10 holders supply at pool creation                   | Float (%)        | Bitquery               | ✅ | |
| 44 | token0_num_holder_launch   | Number of first token holders at the moment of pool creation                                     | Count holders at pool creation                                                       | Integer          | Bitquery               | ✅ | |
| 45 | token1_num_holder_launch   | Number of second token holders at the moment of pool creation                                    | Count holders at pool creation                                                       | Integer          | Bitquery               | ✅ | |
| 46 | token0_liquidity_depth     | Percent of first token total supply locked in pool                                                | balanceOf(pool) at creation / totalSupply at same block *100                         | Float (%)        | RPC                    | ✅ | |
| 47 | token1_liquidity_depth     | Percent of second token total supply locked in pool                                               | balanceOf(pool) at creation / totalSupply at same block *100                         | Float (%)        | RPC                    | ✅ | |
| 48 | label                      | Whether pool contains fraud asset                                                                 | 1 if at least one token is labeled as scam, 0 if both are trustworthy or in https://tokens.uniswap.org/, if 1 truthworthy and other is undefined then take scam_rate | Float          | CSV                    | ✅ | |

## Pseudocode: Single Pool Metrics (batched, cached; one call per metric)

```python
def calculate_metrics_for_token_list(csv_path):
    # Read tokens from categorized_tokens.csv
    tokens = load_categorized_tokens(csv_path)  # returns {token_address -> [name, symbol, label]

    # Calculate scam rate as scams/all
    scam_rate = calculate_scam_rate(tokens)

    # Constants 
    chain_name = "ethereum"
    chain_id = 1
    dex_name = "Uniswap"

    # ---- Step A: Fetch all Uniswap pools for each token (batch Graph queries) ----
    # Uniswap v2: pairs
    v2_pairs_by_token = graph_batch_list_v2_pairs_for_tokens(tokens)   # {token -> [pool_address, pool_creation_timestamp, token0_address, token1_address, token0_name, token1_name, token0_symbol, token1_symbol]}
    # Uniswap v3: pools
    v3_pools_by_token = graph_batch_list_v3_pools_for_tokens(tokens)   # {token -> [pool_address, pool_creation_timestamp, token0_address, token1_address, token0_name, token1_name, token0_symbol, token1_symbol]}


    # Build the global set of pools for processing
    all_pools = list()
    for t in tokens:
        for p in v2_pairs_by_token.get(t, []): all_pools.add(p)
        for p in v3_pools_by_token.get(t, []): all_pools.add(p)

    # Fetch pool reserves in both tokens at the creation block (TODO: test if reserves at creation block are not 0 for each token. if yes then fetch daa for a couple of blocks later)
    token_reserves_by_pool = graph_fetch_token_reserves(all_pools, creations)  # {poolId -> {token_address -> reserve}}

    # Build a set of unique addresses from all collected pools
    unique_addresses = set()
    for pool in all_pools:
        unique_addresses.add(pool['token0_address'])
        unique_addresses.add(pool['token1_address'])

    # Fetch creation timestamp for each unique address from Etherscan
    creations = etherscan_get_contracts_creation(list(unique_addresses))  # {token -> [creation_timestamp, token_deployer]}

    # Fetch token data (name, symbol, totalSupply, etc) for all unique token addresses at token creation block
    token_data = graph_fetch_token_data(list(unique_addresses), creations)  # {token -> [totalSupply, poolAmount]}

    # Fetch first LP data per pool
    mint_data = graph_fetch_first_mint_data(all_pools);  # {pool_address -> [first_pool_activity_time]}

    # Fetch pool deployers from RPC via multicall
    pool_deployers = call_fetch_deployers(pools);  # {pool_address -> deployer}

    # Fetch token owners from RPC via multicall
    token_owners = call_fetch_current_owners(pools);  # {token_address -> owner}

    # Fetch sourcecode for tokens
    sourcecode_responses = etherscan_get_token_sourcecode(unique_addresses);  # {token_address -> [is_proxy, is_verified]}

    # Fetch getCode for deployers and owners, returns true if there is a code
    is_contract = call_fetch_code(pool_deployers, token_owners)  # {token_address -> bool}

    # Fetch tx data from Dune, if ownership is renounced (no owner in list) then fetch data for token deployer from creations
    # If address is a contract then calculate metrics on INCOMING txs, otherwise outgoing
    dune_data = dune_fetch_data(pool_deployers, token_owners, creations, is_contract)  # {address -> [tx_count, address_age, gas_burnt, bytes_deployed, smart_contracts_interacted]}

    # Fetch token holders from bitquery at pool creation block, calculate top_10_holders_percent and count num_holders
    bitquery_data = bitquery_fetch_data(unique_addresses, token_data)   # {address -> [top_10_holders_percent, num_holders]}

    # Fetch tokens from https://tokens.uniswap.org/
    uniswap_verified_tokens = uniswap_web_fetch_data();

    results = []
    for pool in all_pools:
        m = dict()
        m['pool_address'] = pool["pool_address"]
        m['dex_name'] = dex_name
        m['chain_name'] = chain_name
        m['chain_id'] = chain_id
        
        # Set mint data
        mint_info = mint_data.get(pool['pool_address'])
        m['first_pool_activity_time'] = mint_info.get('first_pool_activity_time') if mint_info else None
        
        m['token0_address'] = pool['token0_address']
        m['token1_address'] = pool['token1_address']
        m['token0_name'] = pool['token0_name']
        m['token1_name'] = pool['token1_name']
        m['token0_symbol'] = pool['token0_symbol']
        m['token1_symbol'] = pool['token1_symbol']

        token0_addr = pool['token0_address']
        token1_addr = pool['token1_address']

        m['token0_totalSupply'] = token_data.get(token0_addr, {}).get('totalSupply', None)
        m['token0_poolAmount'] = token_data.get(token0_addr, {}).get('poolAmount', None)

        m['token1_totalSupply'] = token_data.get(token1_addr, {}).get('totalSupply', None)
        m['token1_poolAmount'] = token_data.get(token1_addr, {}).get('poolAmount', None)

        # Calculate token age: time from token contract creation to pool creation
        pool_creation_timestamp = pool.get('pool_creation_timestamp')
        token0_creation_timestamp = creations.get(token0_addr, {}).get('creation_timestamp')
        token1_creation_timestamp = creations.get(token1_addr, {}).get('creation_timestamp')
        
        m['token0_age_minutes'] = (pool_creation_timestamp - token0_creation_timestamp) // 60    
        m['token1_age_minutes'] = (pool_creation_timestamp - token1_creation_timestamp) // 60

        m['token0_is_verified'] = sourcecode_responses.get(token0_addr, {}).get('is_verified', False)
        m['token1_is_verified'] = sourcecode_responses.get(token1_addr, {}).get('is_verified', False)
        m['token0_is_proxy'] = sourcecode_responses.get(token0_addr, {}).get('is_proxy', False)
        m['token1_is_proxy'] = sourcecode_responses.get(token1_addr, {}).get('is_proxy', False)

        # Set ownership renounced metrics for token0 and token1
        token0_owner = token_owners.get(token0_addr)
        token1_owner = token_owners.get(token1_addr)

        m['token0_ownership_renounced'] = (token0_owner is not None and token0_owner.lower() == '0x0000000000000000000000000000000000000000')
        m['token1_ownership_renounced'] = (token1_owner is not None and token1_owner.lower() == '0x0000000000000000000000000000000000000000')

        # ---- Address activity metrics via Dune (with owner -> deployer fallback) ----
        pool_deployer = pool_deployers.get(pool['pool_address'])
        token0_deployer = creations.get(token0_addr, {}).get('token_deployer')
        token1_deployer = creations.get(token1_addr, {}).get('token_deployer')

        token0_owner_or_deployer = token0_owner if token0_owner is not None else token0_deployer
        token1_owner_or_deployer = token1_owner if token1_owner is not None else token1_deployer

        # Tx counts
        m['token0_owner_tx_count'] = dune_data.get(token0_owner_or_deployer, {}).get('tx_count') if token0_owner_or_deployer else None
        m['token1_owner_tx_count'] = dune_data.get(token1_owner_or_deployer, {}).get('tx_count') if token1_owner_or_deployer else None
        m['pool_deployer_tx_count'] = dune_data.get(pool_deployer, {}).get('tx_count') if pool_deployer else None

        # Address ages (minutes)
        m['token0_owner_age'] = dune_data.get(token0_owner_or_deployer, {}).get('address_age') if token0_owner_or_deployer else None
        m['token1_owner_age'] = dune_data.get(token1_owner_or_deployer, {}).get('address_age') if token1_owner_or_deployer else None
        m['pool_deployer_age'] = dune_data.get(pool_deployer, {}).get('address_age') if pool_deployer else None

        # Gas burnt
        m['token0_owner_gas_burnt'] = dune_data.get(token0_owner_or_deployer, {}).get('gas_burnt') if token0_owner_or_deployer else None
        m['token1_owner_gas_burnt'] = dune_data.get(token1_owner_or_deployer, {}).get('gas_burnt') if token1_owner_or_deployer else None
        m['pool_deployer_gas_burnt'] = dune_data.get(pool_deployer, {}).get('gas_burnt') if pool_deployer else None

        # Bytes Deployed
        m['token0_owner_bytes_deployed'] = dune_data.get(token0_owner_or_deployer, {}).get('bytes_deployed') if token0_owner_or_deployer else None
        m['token1_owner_bytes_deployed'] = dune_data.get(token1_owner_or_deployer, {}).get('bytes_deployed') if token1_owner_or_deployer else None
        m['pool_deployer_bytes_deployed'] = dune_data.get(pool_deployer, {}).get('bytes_deployed') if pool_deployer else None

        # Smart contracts interacted
        m['token0_owner_smart_contracts_interacted'] = dune_data.get(token0_owner_or_deployer, {}).get('smart_contracts_interacted') if token0_owner_or_deployer else None
        m['token1_owner_smart_contracts_interacted'] = dune_data.get(token1_owner_or_deployer, {}).get('smart_contracts_interacted') if token1_owner_or_deployer else None
        m['pool_deployer_smart_contracts_interacted'] = dune_data.get(pool_deployer, {}).get('smart_contracts_interacted') if pool_deployer else None
        

        # Set the number of holders
        m['token0_num_holders'] = bitquery_data.get(token0_addr, {}).get('num_holders')
        m['token1_num_holders'] = bitquery_data.get(token1_addr, {}).get('num_holders')

        # Set the top 10 percent holders metrics
        m['token0_top_10_percent'] = bitquery_data.get(token0_addr, {}).get('top_10_percent')
        m['token1_top_10_percent'] = bitquery_data.get(token1_addr, {}).get('top_10_percent')

        # Set liquidity depth as the ratio of pool reserve to totalSupply
        m['token0_liquidity_depth'] = (pool.get('reserve0', 0) / m['token0_totalSupply']) if m.get('token0_totalSupply') else None
        m['token1_liquidity_depth'] = (pool.get('reserve1', 0) / m['token1_totalSupply']) if m.get('token1_totalSupply') else None

        # Label
        m['label'] = calculate_label()

        results.append(m)

    return results
```
