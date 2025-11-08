| id | Feature | Description | Derivation | Type | Source |
|----|----------|--------------|-------------|------|--------|
| 1 | pool_address | Address/Id of the liquidity pool (if any) | Pool address | String | CSV |
| 2 | chain_name | Blockchain (e.g. Ethereum, Solana) | Chain name | String | CSV |
| 3 | chain_id | Numeric or symbolic chain identifier | Chain id | Integer/String | CSV |
| 4 | token0_address | Address of first token in pair | Token address | String | CSV, Graph |
| 5 | token1_address | Address of second token in pair | Token address | String | CSV, Graph |
| 6 | token0_pool_amount | Amount of pools on DEX with first token | Count fetched pools | Integer | Graph |
| 7 | token1_pool_amount | Amount of pools on DEX with second token | Count fetched pools | Integer | Graph |
| 8 | token0_age_minutes | Time from first token contract creation to DEX listing | Token listing timestamp (pool, pair data createdAt field from The Graph) minus token contract creation timestamp (Use Etherscan contract/getcontractcreation to get the creation tx, then fetch the block’s timestamp via a free RPC (e.g., https://cloudflare-eth.com).) | Integer (minutes) | Etherscan, RPC, Graph |
| 9 | token1_age_minutes | Time from first token contract creation to DEX listing | Token listing timestamp (pool, pair data createdAt field from The Graph) minus token contract creation timestamp (Use Etherscan contract/getcontractcreation to get the creation tx, then fetch the block’s timestamp via a free RPC (e.g., https://cloudflare-eth.com).) | Integer (minutes) | Etherscan, RPC, Graph |
| 10 | first_pool_activity_time | Time from DEX listing to first pool LP action | Query Dune with single aggregated SQL query of Mint action, filtered by pool addresses| Integer (minutes) | Dune, Flipside |
| 11 | dex_name | Name of DEX where pool is created | DEX name | String | CSV |
| 12 | token0_max_supply | Maximum supply defined in first token contract | Use batched eth_call at your target block. The fastest way is a Multicall at that block; fall back to direct calls if the multicall contract didn’t exist yet at that block. | Float | RPC |
| 13 | token1_max_supply | Maximum supply defined in second token contract | Use batched eth_call at your target block. The fastest way is a Multicall at that block; fall back to direct calls if the multicall contract didn’t exist yet at that block. | Float | dummy |
| 14 | token0_minting_enabled | Can the first token be minted after deployment? | Check smart contract code | Boolean | dummy |
| 15 | token1_minting_enabled | Can the second token be minted after deployment? | Check smart contract code | Boolean | dummy |
| 16 | token0_verified_contract | Is the first token contract verified on-chain? | Call getsourcecode endpoint for proxy and for implementation | Boolean | Etherscan |
| 17 | token1_verified_contract | Is the second token contract verified on-chain? | Call getsourcecode endpoint for proxy and for implementation | Boolean | Etherscan |
| 18 | token0_ownership_renounced | Is there an owner of a first token contract? | Value specified in contract | Boolean | dummy |
| 19 | token1_ownership_renounced | Is there an owner of a second token contract? | Value specified in contract | Boolean | dummy |
| 20 | token0_proxy | Whether the first token uses a proxy | Call getsourcecode endpoint for proxy | Boolean | Etherscan |
| 21 | token0_pausable | If the first token has pausable functionality | Check code | Boolean | dummy |
| 22 | token0_burning_enabled | Can the first token be burned after deployment? | Check code | Boolean | dummy |
| 23 | token1_proxy | Whether the second token uses a proxy | Call getsourcecode endpoint for proxy | Boolean | Etherscan |
| 24 | token1_pausable | If the second token has pausable functionality | Check code | Boolean | dummy |
| 25 | token1_burning_enabled | Can the second token be burned after deployment? | Check code | Boolean | dummy |
| 26 | token0_owner_tx_count | Amount of transactions from the owner address | Count txs | Integer | dummy |
| 27 | token1_owner_tx_count | Amount of transactions from the owner address | Count txs | Integer | dummy |
| 28 | pool_deployer_tx_count | Amount of transactions from the pool deployer address | Count txs | Integer | dummy |
| 29 | token0_owner_age | Time passed from first token owner address first tx | Current timestamp minus tx timestamp | Integer (minutes) | dummy |
| 30 | token1_owner_age | Time passed from second token owner address first tx | Current timestamp minus tx timestamp | Integer (minutes) | dummy |
| 31 | pool_deployer_age | Time passed from pool deployer address first tx | Current timestamp minus tx timestamp | Integer (minutes) | dummy |
| 32 | token0_owner_gas_burnt | Amount of gas burnt by first token owner address | Sum up all gas burnt by txs | Float | dummy |
| 33 | token1_owner_gas_burnt | Amount of gas burnt by second token owner address | Sum up all gas burnt by txs | Float | dummy |
| 34 | pool_deployer_gas_burnt | Amount of gas burnt by pool deployer address | Sum up all gas burnt by txs | Float | dummy |
| 35 | token0_top_10_holders_percent | Supply of first token owned by top 10 holders in percent | Calculate percent of summed top 10 holders supply at the moment of pool creation | Float (%) | dummy |
| 36 | token1_top_10_holders_percent | Supply of second token owned by top 10 holders in percent | Calculate percent of summed top 10 holders supply at the moment of pool creation | Float (%) | dummy |
| 37 | token0_num_holder_launch | Number of first token holders at the moment of launch | Count holders at the moment pool is created | Integer | dummy |
| 38 | token1_num_holder_launch | Number of second token holders at the moment of launch | Count holders at the moment pool is created | Integer | dummy |
| 39 | token0_liquidity_depth | Percent of first token total supply locked in pool | Calculate percent based on amount of tokens in pool at launch vs initial supply | Float (%) | dummy |
| 40 | token1_liquidity_depth | Percent of second token total supply locked in pool | Calculate percent based on amount of tokens in pool at launch vs initial supply | Float (%) | dummy |
| 41 | label | Whether pool contains fraud asset | True if one token is labeled as scam, false if both are trustworthy, undefined otherwise | Boolean/Enum | dummy |
