from __future__ import annotations

import os
from dotenv import load_dotenv
import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime


import requests  # HTTP client for The Graph, Etherscan, Dune, Bitquery, etc.
# from web3 import Web3  # Uncomment when wiring up on-chain RPC calls


logger = logging.getLogger(__name__)
load_dotenv()
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")
UNISWAP_V2_SUBGRAPH_ID = "A3Np3RQbaBA6oKJgiwDJeo5T3zrYfGHPWFYayMwtNDum"
UNISWAP_V3_SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
UNISWAP_V2_ENDPOINT = f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/{UNISWAP_V2_SUBGRAPH_ID}"
UNISWAP_V3_ENDPOINT =f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/{UNISWAP_V3_SUBGRAPH_ID}"
THE_GRAPH_API_BATCH_SIZE = 1000  # Max entities per query
TOKEN_ADDRESS_PROCESSING_BATCH_SIZE = 300 # How many token addresses to include in a single GraphQL query

def log_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that applies standard logging around a function call.

    - Logs function entry with argument summary.
    - Logs execution duration on success.
    - Logs full stack trace on exception and re-raises.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        arg_preview = ", ".join(
            [
                *(repr(a) for a in args[:3]),
                *(f"{k}={v!r}" for k, v in list(kwargs.items())[:3]),
            ]
        )
        logger.info("Calling %s(%s)", func.__name__, arg_preview)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception:
            logger.exception("Error in %s", func.__name__)
            raise
        else:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.info("Finished %s in %.2f ms", func.__name__, duration_ms)
            return result

    return wrapper

def _save_dict_to_json(
    data: Dict, output_dir: Path, base_filename: str
) -> None:
    """Saves a dictionary to a timestamped JSON file in the specified directory."""
    try:
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{timestamp}.json"
        filepath = output_dir / filename

        logger.info("Saving data to %s", filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            # Use indent for a pretty-printed, human-readable file
            json.dump(data, f, indent=4)
            
    except (IOError, TypeError) as e:
        logger.exception("Failed to save data to JSON file: %s", e)


# =========================
# Data models / type hints
# =========================


@dataclass
class TokenRecord:
    """Single row from categorized_tokens.csv."""

    address: str
    name: str
    symbol: str
    label: bool  # e.g. True for scam, False for trustworthy, etc.


@dataclass
class PoolInfo:
    """Basic metadata for a Uniswap V2/V3 pool."""

    pool_address: str
    pool_creation_timestamp: int  # unix seconds
    token0_address: str
    token1_address: str
    token0_name: str
    token1_name: str
    token0_symbol: str
    token1_symbol: str
    # Optional reserve fields (can be filled later)
    reserve0: Optional[int] = None
    reserve1: Optional[int] = None


@dataclass
class TokenCreationInfo:
    """Creation metadata for a token contract."""

    creation_timestamp: int  # unix seconds
    token_deployer: str


@dataclass
class TokenOnchainData:
    """On-chain token data at or around pool creation."""

    total_supply: Optional[float]
    pool_amount: Optional[int]


@dataclass
class MintInfo:
    """First LP event information per pool."""

    first_pool_activity_time: int  # minutes from listing to first LP action


@dataclass
class SourceCodeInfo:
    """Etherscan-style source code metadata for a contract."""

    is_proxy: bool
    is_verified: bool


@dataclass
class AddressActivity:
    """Aggregated address-level metrics from Dune."""

    tx_count: Optional[int]
    address_age: Optional[int]  # minutes since first tx
    gas_burnt: Optional[int]
    bytes_deployed: Optional[int]
    smart_contracts_interacted: Optional[int]


@dataclass
class HolderDistribution:
    """Token holder distribution metrics from Bitquery."""

    top_10_percent: Optional[float]
    num_holders: Optional[int]


@dataclass
class PoolMetricsRow:
    """Final, flattened row of metrics for a single pool."""

    pool_address: str
    dex_name: str
    chain_name: str
    chain_id: int

    first_pool_activity_time: Optional[int]

    token0_address: str
    token1_address: str
    token0_name: str
    token1_name: str
    token0_symbol: str
    token1_symbol: str

    token0_total_supply: Optional[float]
    token0_pool_amount: Optional[int]
    token1_total_supply: Optional[float]
    token1_pool_amount: Optional[int]

    token0_age_minutes: Optional[int]
    token1_age_minutes: Optional[int]

    token0_verified_contract: bool
    token1_verified_contract: bool
    token0_proxy: bool
    token1_proxy: bool

    token0_ownership_renounced: bool
    token1_ownership_renounced: bool

    token0_owner_tx_count: Optional[int]
    token1_owner_tx_count: Optional[int]
    pool_deployer_tx_count: Optional[int]

    token0_owner_age: Optional[int]
    token1_owner_age: Optional[int]
    pool_deployer_age: Optional[int]

    token0_owner_gas_burnt: Optional[int]
    token1_owner_gas_burnt: Optional[int]
    pool_deployer_gas_burnt: Optional[int]

    token0_owner_bytes_deployed: Optional[int]
    token1_owner_bytes_deployed: Optional[int]
    pool_deployer_bytes_deployed: Optional[int]

    token0_owner_smart_contracts_interacted: Optional[int]
    token1_owner_smart_contracts_interacted: Optional[int]
    pool_deployer_smart_contracts_interacted: Optional[int]

    token0_num_holders: Optional[int]
    token1_num_holders: Optional[int]
    token0_top_10_percent: Optional[float]
    token1_top_10_percent: Optional[float]

    token0_liquidity_depth: Optional[float]
    token1_liquidity_depth: Optional[float]

    label: float


TokensByAddress = Dict[str, TokenRecord]
PoolsByToken = Dict[str, List[PoolInfo]]
CreationMap = Dict[str, TokenCreationInfo]
TokenDataMap = Dict[str, TokenOnchainData]
MintDataMap = Dict[str, MintInfo]
SourceCodeMap = Dict[str, SourceCodeInfo]
AddressActivityMap = Dict[str, AddressActivity]
HolderDistributionMap = Dict[str, HolderDistribution]
OwnerMap = Dict[str, Optional[str]]
IsContractMap = Dict[str, bool]
ReservesByPool = Dict[str, Dict[str, int]]


# =========================
# CSV loading & basic utils
# =========================


@log_call
def load_categorized_tokens(csv_path: Path) -> TokensByAddress:
    """
    Read tokens from categorized_tokens.csv.

    Expected columns (at minimum): address, name, symbol, label.
    """
    # TODO: Implement CSV parsing logic that returns TokensByAddress keyed by token address.
    tokens: TokensByAddress = {}
    try:
        with open(csv_path, mode="r", encoding="utf-8") as infile:
            # Using DictReader to easily access columns by name
            reader = csv.DictReader(infile)
            for row in reader:
                # Normalize the address to lowercase for consistent keying
                address = row["address"].lower()
                
                # Convert the label to boolean. True if the label is 'scam'.
                is_scam = row["label"].strip().lower() == "scam"

                # Create a TokenRecord instance and add it to the dictionary
                tokens[address] = TokenRecord(
                    address=address,
                    name=row["name"],
                    symbol=row["symbol"],
                    label=is_scam,
                )
    except FileNotFoundError:
        logger.error("Input CSV file not found at: %s", csv_path)
        raise
    except KeyError as e:
        logger.error(
            "The CSV file %s is missing the required column: %s", csv_path, e
        )
        raise
    
    return tokens


@log_call
def calculate_scam_rate(tokens: TokensByAddress) -> float:
    """
    Calculate scam rate as scams / all.

    A 'scam' is typically identified via the TokenRecord.label field.
    """
    # TODO: Implement scam_rate computation based on token labels.
    if not tokens:
        logger.warning("Cannot calculate scam rate: the token dictionary is empty.")
        return 0.0

    # Count tokens where the 'label' is True (indicating a scam)
    scam_count = sum(1 for token in tokens.values() if token.label)
    
    total_tokens = len(tokens)

    # Return the ratio of scams to the total number of tokens
    return scam_count / total_tokens


# =========================
# External data fetchers
# =========================


@log_call
def graph_batch_list_v2_pairs_for_tokens(
    tokens: TokensByAddress, save_to_dir: Optional[Path] = None
) -> PoolsByToken:
    """
    Query Uniswap V2 pairs from The Graph for all tokens using the gateway.

    Performs batched GraphQL queries and returns pools grouped by the input token address.
    If save_to_dir is provided, the results are saved to a timestamped JSON file.
    """
    
    if not tokens:
        return {}

    pools_by_token: Dict[str, List[PoolInfo]] = defaultdict(list)
    token_addresses = list(tokens.keys())

    for i in range(0, len(token_addresses), TOKEN_ADDRESS_PROCESSING_BATCH_SIZE):
        address_batch = token_addresses[i : i + TOKEN_ADDRESS_PROCESSING_BATCH_SIZE]
        all_pairs_for_batch: List[Dict] = []
        skip = 0
        while True:
            query = """
            query ($addresses: [String!], $skip: Int!) {
              pairs(
                first: %d,
                skip: $skip,
                where: {
                  or: [
                    { token0_in: $addresses },
                    { token1_in: $addresses }
                  ]
                }
              ) {
                id
                createdAtTimestamp
                token0 { id name symbol }
                token1 { id name symbol }
              }
            }
            """ % THE_GRAPH_API_BATCH_SIZE

            try:
                response = requests.post(
                    UNISWAP_V2_ENDPOINT,
                    json={
                        "query": query,
                        "variables": {"addresses": address_batch, "skip": skip},
                    },
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error("Failed to query Uniswap V2 subgraph: %s", e)
                break

            data = response.json()
            if "errors" in data:
                logger.error("GraphQL query returned errors: %s", data["errors"])
                break

            pairs = data.get("data", {}).get("pairs", [])
            all_pairs_for_batch.extend(pairs)

            if len(pairs) < THE_GRAPH_API_BATCH_SIZE:
                break
            skip += THE_GRAPH_API_BATCH_SIZE

        address_batch_set = set(a.lower() for a in address_batch)
        for pair in all_pairs_for_batch:
            pool_info = PoolInfo(
                pool_address=pair["id"],
                pool_creation_timestamp=int(pair["createdAtTimestamp"]),
                token0_address=pair["token0"]["id"],
                token1_address=pair["token1"]["id"],
                token0_name=pair["token0"]["name"],
                token1_name=pair["token1"]["name"],
                token0_symbol=pair["token0"]["symbol"],
                token1_symbol=pair["token1"]["symbol"],
            )
            if pool_info.token0_address in address_batch_set:
                pools_by_token[pool_info.token0_address].append(pool_info)
            if pool_info.token1_address in address_batch_set:
                pools_by_token[pool_info.token1_address].append(pool_info)

    if save_to_dir:
        serializable_data = {
            token: [asdict(pool) for pool in pools]
            for token, pools in pools_by_token.items()
        }
        _save_dict_to_json(serializable_data, save_to_dir, "uniswap_v2_pairs")

    return dict(pools_by_token)


@log_call
def graph_batch_list_v3_pools_for_tokens(
    tokens: TokensByAddress, save_to_dir: Optional[Path] = None
) -> PoolsByToken:
    """
    Query Uniswap V3 pools from The Graph for all tokens using the gateway.

    Performs batched GraphQL queries and returns pools grouped by the input token address.
    If save_to_dir is provided, the results are saved to a timestamped JSON file.
    """
    if not tokens:
        return {}

    pools_by_token: Dict[str, List[PoolInfo]] = defaultdict(list)
    token_addresses = list(tokens.keys())

    # Process token addresses in manageable batches
    for i in range(0, len(token_addresses), TOKEN_ADDRESS_PROCESSING_BATCH_SIZE):
        address_batch = token_addresses[i : i + TOKEN_ADDRESS_PROCESSING_BATCH_SIZE]
        all_pools_for_batch: List[Dict] = []
        skip = 0
        while True:
            # The query is identical to V2, just querying "pools" instead of "pairs"
            query = """
            query ($addresses: [String!], $skip: Int!) {
              pools(
                first: %d,
                skip: $skip,
                where: {
                  or: [
                    { token0_in: $addresses },
                    { token1_in: $addresses }
                  ]
                }
              ) {
                id
                createdAtTimestamp
                token0 { id name symbol }
                token1 { id name symbol }
              }
            }
            """ % THE_GRAPH_API_BATCH_SIZE

            try:
                response = requests.post(
                    UNISWAP_V3_ENDPOINT,
                    json={
                        "query": query,
                        "variables": {"addresses": address_batch, "skip": skip},
                    },
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error("Failed to query Uniswap V3 subgraph: %s", e)
                break

            data = response.json()
            if "errors" in data:
                logger.error("GraphQL query returned errors: %s", data["errors"])
                break

            # Parse "pools" from the response
            pools = data.get("data", {}).get("pools", [])
            all_pools_for_batch.extend(pools)

            if len(pools) < THE_GRAPH_API_BATCH_SIZE:
                break
            skip += THE_GRAPH_API_BATCH_SIZE

        # Map the fetched pool data back to the original input tokens
        address_batch_set = set(a.lower() for a in address_batch)
        for pool in all_pools_for_batch:
            pool_info = PoolInfo(
                pool_address=pool["id"],
                pool_creation_timestamp=int(pool["createdAtTimestamp"]),
                token0_address=pool["token0"]["id"],
                token1_address=pool["token1"]["id"],
                token0_name=pool["token0"]["name"],
                token1_name=pool["token1"]["name"],
                token0_symbol=pool["token0"]["symbol"],
                token1_symbol=pool["token1"]["symbol"],
            )
            if pool_info.token0_address in address_batch_set:
                pools_by_token[pool_info.token0_address].append(pool_info)
            if pool_info.token1_address in address_batch_set:
                pools_by_token[pool_info.token1_address].append(pool_info)

    if save_to_dir:
        serializable_data = {
            token: [asdict(pool) for pool in pools]
            for token, pools in pools_by_token.items()
        }
        # Use a different filename for the V3 cache
        _save_dict_to_json(serializable_data, save_to_dir, "uniswap_v3_pools")

    return dict(pools_by_token)


@log_call
def graph_fetch_token_reserves(all_pools: Sequence[PoolInfo]) -> ReservesByPool:
    """
    Fetch token reserves for each pool at the pool creation block.

    The result should be {pool_address -> {token_address -> reserve_int}}.
    """
    # TODO: Implement Graph queries (or RPC calls) to fetch reserves at creation.
    raise NotImplementedError


@log_call
def etherscan_get_contracts_creation(
    token_addresses: Sequence[str],
) -> CreationMap:
    """
    Fetch creation timestamp and deployer for each token contract from Etherscan.

    Returns a mapping from token address to TokenCreationInfo.
    """
    # TODO: Implement Etherscan API calls (get_contract_creation or equivalent).
    raise NotImplementedError


@log_call
def graph_fetch_token_data(
    token_addresses: Sequence[str],
    creations: CreationMap,
) -> TokenDataMap:
    """
    Fetch token data (totalSupply, poolAmount, etc.) for all tokens.

    Should query at or near the token creation / pool creation block using Graph/RPC.
    """
    # TODO: Implement token-level data fetching via The Graph or direct RPC.
    raise NotImplementedError


@log_call
def graph_fetch_first_mint_data(all_pools: Sequence[PoolInfo]) -> MintDataMap:
    """
    Fetch first LP (mint) data per pool from The Graph.

    Should capture time from listing to first LP action in minutes.
    """
    # TODO: Implement query for first mint events per pool maybe with separate helper function for V2/V3.
    raise NotImplementedError


@log_call
def call_fetch_deployers(all_pools: Sequence[PoolInfo]) -> Dict[str, str]:
    """
    Fetch pool deployer addresses via RPC (e.g., by inspecting transaction / logs).

    Returns {pool_address -> deployer_address}.
    """
    # TODO: Implement multicall / RPC logic to resolve deployers for each pool.
    raise NotImplementedError


@log_call
def call_fetch_current_owners(token_addresses: Sequence[str]) -> OwnerMap:
    """
    Fetch current owner for each token contract via RPC.

    Returns {token_address -> owner_address or None}.
    """
    # TODO: Implement owner() calls (possibly via multicall) for token contracts.
    raise NotImplementedError


@log_call
def etherscan_get_token_sourcecode(
    token_addresses: Sequence[str],
) -> SourceCodeMap:
    """
    Fetch source code metadata (is_proxy, is_verified) for each token from Etherscan.
    """
    # TODO: Implement Etherscan getsourcecode endpoint usage.
    raise NotImplementedError


@log_call
def call_fetch_code(
    pool_deployers: Mapping[str, str],
    token_owners: OwnerMap,
) -> IsContractMap:
    """
    Fetch code for deployers and owners via RPC to determine if an address is a contract.

    Returns {address -> bool_is_contract}.
    """
    # TODO: Implement eth_getCode calls (possibly via multicall) for all relevant addresses.
    raise NotImplementedError


@log_call
def dune_fetch_data(
    pool_deployers: Mapping[str, str],
    token_owners: OwnerMap,
    creations: CreationMap,
    is_contract: IsContractMap,
) -> AddressActivityMap:
    """
    Fetch aggregated tx data from Dune for each relevant address.

    If ownership is renounced (no owner in list), fall back to token deployer from creations.
    For contract addresses, compute metrics on INCOMING txs; otherwise on outgoing.
    """
    # TODO: Implement Dune SQL/API queries and transform into AddressActivityMap.
    raise NotImplementedError


@log_call
def bitquery_fetch_data(
    token_addresses: Sequence[str],
    token_data: TokenDataMap,
) -> HolderDistributionMap:
    """
    Fetch token holder statistics from Bitquery at pool creation block.

    Should compute top_10_percent and num_holders per token.
    """
    # TODO: Implement Bitquery GraphQL/API queries for holder distributions.
    raise NotImplementedError


@log_call
def uniswap_web_fetch_data() -> Set[str]:
    """
    Fetch the set of verified tokens from https://tokens.uniswap.org/.

    Returns a set of token addresses that are considered trustworthy/verified.
    """
    # TODO: Implement HTTP fetch and JSON parsing of Uniswap token list.
    raise NotImplementedError


# =========================
# Labeling & metric helpers
# =========================


@log_call
def calculate_label(
    token0_address: str,
    token1_address: str,
    tokens: TokensByAddress,
    scam_rate: float,
    uniswap_verified_tokens: Set[str],
) -> float:
    """
    Compute the pool label based on token labels and Uniswap verification.

    Suggested behavior (to match the spec in static_pool_metrics.md):
      - 1 if at least one token is labeled as scam.
      - 0 if both tokens are trustworthy or appear in the Uniswap token list.
      - If one is trustworthy and the other is undefined, use scam_rate as a soft label.
    """
    # TODO: Implement labeling logic based on token metadata and scam_rate.
    raise NotImplementedError


@log_call
def build_unique_token_addresses(all_pools: Sequence[PoolInfo]) -> List[str]:
    """
    Extract a deduplicated list of token addresses from the full pool list.
    """
    # TODO: Implement simple extraction of token0/token1 addresses into a unique list.
    raise NotImplementedError


@log_call
def build_pool_metrics_row(
    pool: PoolInfo,
    chain_name: str,
    chain_id: int,
    dex_name: str,
    scam_rate: float,
    tokens: TokensByAddress,
    creations: CreationMap,
    token_data: TokenDataMap,
    mint_data: MintDataMap,
    pool_deployers: Mapping[str, str],
    token_owners: OwnerMap,
    sourcecode_responses: SourceCodeMap,
    dune_data: AddressActivityMap,
    bitquery_data: HolderDistributionMap,
    reserves_by_pool: ReservesByPool,
    uniswap_verified_tokens: Set[str],
) -> PoolMetricsRow:
    """
    Assemble a fully-populated PoolMetricsRow for a single pool using all
    pre-fetched data maps.
    """
    # TODO: Implement transformation logic mirroring the pseudocode in static_pool_metrics.md.
    raise NotImplementedError


# =========================
# Main orchestration
# =========================


@log_call
def calculate_metrics_for_token_list(csv_path: Path) -> List[PoolMetricsRow]:
    """
    High-level orchestration for calculating metrics for the token list.

    This closely follows the pseudocode in static_pool_metrics.md: batch & cache
    all external calls, then build one metrics row per pool.
    """
    # Load tokens from categorized_tokens.csv
    tokens = load_categorized_tokens(csv_path)

    # Calculate scam rate as scams / all
    scam_rate = calculate_scam_rate(tokens)

    # Constants
    chain_name = "ethereum"
    chain_id = 1
    dex_name = "Uniswap"

    # ---- Step A: Fetch all Uniswap pools for each token (batch Graph queries) ----
    v2_pairs_by_token = graph_batch_list_v2_pairs_for_tokens(tokens)
    v3_pools_by_token = graph_batch_list_v3_pools_for_tokens(tokens)

    # Build the global list of pools for processing
    all_pools: List[PoolInfo] = []
    for t_addr, pools in v2_pairs_by_token.items():
        all_pools.extend(pools)
    for t_addr, pools in v3_pools_by_token.items():
        all_pools.extend(pools)

    # Build a deduplicated set of unique token addresses from all pools
    unique_token_addresses = build_unique_token_addresses(all_pools)

    # Fetch creation timestamp and deployer for each unique token address
    creations = etherscan_get_contracts_creation(unique_token_addresses)

    # Fetch token data (name, symbol, totalSupply, poolAmount, etc.)
    token_data = graph_fetch_token_data(unique_token_addresses, creations)

    # Fetch first LP data per pool
    mint_data = graph_fetch_first_mint_data(all_pools)

    # Fetch pool deployers from RPC via multicall
    pool_deployers = call_fetch_deployers(all_pools)

    # Fetch token owners from RPC via multicall
    token_owners = call_fetch_current_owners(unique_token_addresses)

    # Fetch source code metadata for tokens
    sourcecode_responses = etherscan_get_token_sourcecode(unique_token_addresses)

    # Fetch getCode for deployers and owners, returns True if there is code
    is_contract = call_fetch_code(pool_deployers, token_owners)

    # Fetch tx data from Dune
    dune_data = dune_fetch_data(pool_deployers, token_owners, creations, is_contract)

    # Fetch token holders from Bitquery at pool creation block
    bitquery_data = bitquery_fetch_data(unique_token_addresses, token_data)

    # Fetch pool reserves in both tokens at the creation block
    reserves_by_pool = graph_fetch_token_reserves(all_pools)

    # Fetch tokens from https://tokens.uniswap.org/
    uniswap_verified_tokens = uniswap_web_fetch_data()

    # Assemble metrics per pool
    results: List[PoolMetricsRow] = []
    for pool in all_pools:
        metrics_row = build_pool_metrics_row(
            pool=pool,
            chain_name=chain_name,
            chain_id=chain_id,
            dex_name=dex_name,
            scam_rate=scam_rate,
            tokens=tokens,
            creations=creations,
            token_data=token_data,
            mint_data=mint_data,
            pool_deployers=pool_deployers,
            token_owners=token_owners,
            sourcecode_responses=sourcecode_responses,
            dune_data=dune_data,
            bitquery_data=bitquery_data,
            reserves_by_pool=reserves_by_pool,
            uniswap_verified_tokens=uniswap_verified_tokens,
        )
        results.append(metrics_row)

    return results


@log_call
def write_metrics_to_csv(rows: Sequence[PoolMetricsRow], output_path: Path) -> None:
    """
    Serialize the final list of PoolMetricsRow objects into a CSV file.
    """
    # TODO: Implement CSV writing using csv.DictWriter over dataclasses.asdict(row).
    raise NotImplementedError


@log_call
def main() -> None:
    """
    CLI entrypoint: load input CSV, compute metrics, and write them to disk.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


    #graph_batch_list_v2_pairs_for_tokens(load_categorized_tokens(Path("categorized_tokens_orig.csv")), Path("output"))
    #graph_batch_list_v3_pools_for_tokens(load_categorized_tokens(Path("categorized_tokens_orig.csv")), Path("output"))


if __name__ == "__main__":
    main()


