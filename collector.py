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
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
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
    - Logs function entry with a summarized argument preview.
    - Logs execution duration on success.
    - Logs full stack trace on exception and re-raises.
    """

    def _preview_arg(arg: Any) -> str:
        """Create a sensible, summarized preview of a function argument."""
        # For common large collections, show their type and size instead of content.
        if isinstance(arg, (list, tuple, set)):
            return f"<{type(arg).__name__} len={len(arg)}>"
        if isinstance(arg, dict):
            return f"<dict len={len(arg)}>"
        
        # For other types, use the default representation but truncate if it's too long.
        s = repr(arg)
        return s if len(s) < 120 else s[:117] + "..."

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Use the new helper to create a clean preview for all arguments.
        arg_preview = ", ".join(
            [
                *(_preview_arg(a) for a in args),
                *(f"{k}={_preview_arg(v)}" for k, v in kwargs.items()),
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

# =========================
# Generic caching helpers
# =========================
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

def _find_latest_cache_file(cache_dir: Path, base_filename: str) -> Optional[Path]:
    """Finds the most recent timestamped cache file in a directory."""
    latest_file = None
    latest_datetime = datetime.min
    cache_dir.mkdir(parents=True, exist_ok=True)

    for f in cache_dir.glob(f"{base_filename}_*.json"):
        try:
            parts = f.stem.split('_')
            datetime_str = f"{parts[-2]}_{parts[-1]}"
            file_datetime = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
            if file_datetime > latest_datetime:
                latest_datetime = file_datetime
                latest_file = f
        except (ValueError, IndexError):
            continue
    return latest_file

def _load_or_fetch_data(
    cache_dir: Path,
    base_filename: str,
    fetch_function: Callable[..., Any],
    reconstruct_class: Optional[type] = None,
    **fetch_kwargs: Any,
) -> Any:
    """
    Generic function to check for a cached JSON file. If found, loads it. 
    Otherwise, runs the fetch_function to get the data and saves it.
    """
    # 1. Call the helper to find the latest file.
    latest_cache_file = _find_latest_cache_file(cache_dir, base_filename)

    if latest_cache_file:
        logger.info("Found cache file. Loading data from: %s", latest_cache_file)
        with open(latest_cache_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            
            # If a reconstruction class is provided, rebuild the objects
            if reconstruct_class:
                # This handles both list-of-dicts and dict-of-dicts-of-lists
                if isinstance(loaded_data, dict) and all(isinstance(v, list) for v in loaded_data.values()):
                     reconstructed_data = defaultdict(list)
                     for key, item_list in loaded_data.items():
                         for item_dict in item_list:
                             reconstructed_data[key].append(reconstruct_class(**item_dict))
                     return dict(reconstructed_data)
                elif isinstance(loaded_data, dict):
                    return {key: reconstruct_class(**value) for key, value in loaded_data.items()}

            return loaded_data # Return raw dicts if no reconstruction needed
    else:
        logger.info("No cache file found for '%s'. Fetching from API...", base_filename)
        data_to_cache = fetch_function(**fetch_kwargs)
        
        # The save function needs the data in dict format, so we convert it if necessary
        if reconstruct_class:
            if isinstance(data_to_cache, dict) and all(isinstance(v, list) for v in data_to_cache.values()):
                serializable_data = {
                    token: [asdict(item) for item in items]
                    for token, items in data_to_cache.items()
                }
                _save_dict_to_json(serializable_data, cache_dir, base_filename)
            elif isinstance(data_to_cache, dict):
                serializable_data = {key: asdict(value) for key, value in data_to_cache.items()}
                _save_dict_to_json(serializable_data, cache_dir, base_filename)
        else:
            _save_dict_to_json(data_to_cache, cache_dir, base_filename)
            
        return data_to_cache

# =========================
# Labeling / metric helpers
# =========================



@log_call
def _process_mints_for_mint_data_map(
    first_mints: Dict[str, Dict[str, Any]],
    all_pools: Sequence[PoolInfo],
    save_to_dir: Optional[Path] = None,
) -> MintDataMap:
    """Transforms raw first mint data into the MintDataMap format and saves it."""
    # NO loading logic. Just process and save.
    pool_creation_map = {p.pool_address: p.pool_creation_timestamp for p in all_pools}
    mint_data_map: MintDataMap = {}

    for pool_address, mint_info in first_mints.items():
        pool_creation_ts = pool_creation_map.get(pool_address)
        if pool_creation_ts:
            delta_seconds = max(0, mint_info["timestamp"] - pool_creation_ts)
            mint_data_map[pool_address] = MintInfo(
                first_pool_activity_time=int(delta_seconds / 60)
            )

    if save_to_dir:
        serializable_data = {
            addr: asdict(info) for addr, info in mint_data_map.items()
        }
        _save_dict_to_json(serializable_data, save_to_dir, "first_mint_data")

    return mint_data_map

@log_call
def _process_mints_for_reserves_by_pool(
    first_mints: Dict[str, Dict[str, Any]],
    save_to_dir: Optional[Path] = None,
) -> ReservesByPool:
    """Transforms raw first mint data into the ReservesByPool format and saves it."""
    # NO loading logic. Just process and save.
    reserves_by_pool: ReservesByPool = {}
    for pool_address, mint_info in first_mints.items():
        try:
            reserves_by_pool[pool_address] = {
                mint_info["token0_address"]: int(float(mint_info["amount0"])),
                mint_info["token1_address"]: int(float(mint_info["amount1"])),
            }
        except (ValueError, KeyError) as e:
            logger.warning(
                "Could not parse reserves for pool %s from mint data: %s", pool_address, e
            )

    if save_to_dir:
        _save_dict_to_json(reserves_by_pool, save_to_dir, "initial_token_reserves")

    return reserves_by_pool


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
def graph_batch_list_v2_pairs_for_tokens(tokens: TokensByAddress) -> PoolsByToken:
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

    return dict(pools_by_token)


@log_call
def graph_batch_list_v3_pools_for_tokens(tokens: TokensByAddress) -> PoolsByToken:
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

    return dict(pools_by_token)

# Implementation for this function and graph_fetch_first_mint_data are similar, so we will fetch data for both in one go.
# @log_call
# def graph_fetch_token_reserves(all_pools: Sequence[PoolInfo]) -> ReservesByPool:
#     """
#     Fetch token reserves for each pool at the pool creation block.

#     The result should be {pool_address -> {token_address -> reserve_int}}.
#     """
#     # TODO: Implement Graph queries (or RPC calls) to fetch reserves at creation.
#     raise NotImplementedError


@log_call
def etherscan_get_contracts_creation(
    token_addresses: Sequence[str],
) -> CreationMap:
    """
    Fetch creation timestamp and deployer for each token contract from Etherscan.

    Returns a mapping from token address to TokenCreationInfo.
    """
    # TODO: Implement Etherscan API calls (get_contract_creation or equivalent).
    """[STUB] Returns an empty map. To be implemented."""
    logger.warning("etherscan_get_contracts_creation is not implemented. Returning empty data.")
    return {}
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
    """[STUB] Returns an empty map. To be implemented."""
    logger.warning("graph_fetch_token_data is not implemented. Returning empty data.")
    return {}

def _fetch_first_mints_for_pools(
    all_pools: Sequence[PoolInfo],
) -> Dict[str, Dict[str, Any]]:
    """
    Private helper to fetch the single earliest mint event for a list of pools.
    This version includes a robust retry mechanism to handle network errors.
    """
    if not all_pools:
        return {}

    pool_info_map = {p.pool_address: p for p in all_pools}
    all_pool_addresses = list(pool_info_map.keys())
    
    first_mints = {}


    # This will automatically retry on connection errors and common server errors (5xx)
    retry_strategy = Retry(
        total=5,  # Total number of retries
        backoff_factor=2,  # Exponential backoff (e.g., 2s, 4s, 8s...)
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"] # Add POST to the list of allowed methods
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)

    #  Query V2 Subgraph 
    logger.info("Fetching first mints from Uniswap V2 subgraph...")
    for i in range(0, len(all_pool_addresses), TOKEN_ADDRESS_PROCESSING_BATCH_SIZE):
        address_batch = all_pool_addresses[i : i + TOKEN_ADDRESS_PROCESSING_BATCH_SIZE]
        query = """
        query ($addresses: [String!]) {
          mints(first: 1000, orderBy: timestamp, orderDirection: asc, where: { pair_in: $addresses }) {
            timestamp, amount0, amount1, pair { id }
          }
        }
        """
        try:
            # Use the session object instead of requests directly
            response = session.post(
                UNISWAP_V2_ENDPOINT,
                json={"query": query, "variables": {"addresses": address_batch}},
                timeout=45,  # Increased timeout slightly for complex queries
            )
            response.raise_for_status()
            data = response.json().get("data", {}).get("mints", [])
            for mint in data:
                pool_id = mint["pair"]["id"]
                if pool_id not in first_mints:
                    first_mints[pool_id] = mint
        except requests.RequestException as e:
            # This will now only log an error after all retries have failed
            logger.error("Failed to fetch V2 mints for batch starting at index %d after multiple retries: %s", i, e)
        except Exception as e:
            logger.error("A non-network error occurred during V2 mint fetching: %s", e)

    # Query V3 Subgraph 
    logger.info("Fetching first mints from Uniswap V3 subgraph...")
    for i in range(0, len(all_pool_addresses), TOKEN_ADDRESS_PROCESSING_BATCH_SIZE):
        address_batch = all_pool_addresses[i : i + TOKEN_ADDRESS_PROCESSING_BATCH_SIZE]
        query = """
        query ($addresses: [String!]) {
          mints(first: 1000, orderBy: timestamp, orderDirection: asc, where: { pool_in: $addresses }) {
            timestamp, amount0, amount1, pool { id }
          }
        }
        """
        try:
            # Use the same session object
            response = session.post(
                UNISWAP_V3_ENDPOINT,
                json={"query": query, "variables": {"addresses": address_batch}},
                timeout=45,
            )
            response.raise_for_status()
            data = response.json().get("data", {}).get("mints", [])
            for mint in data:
                pool_id = mint["pool"]["id"]
                if pool_id not in first_mints:
                    first_mints[pool_id] = mint
        except requests.RequestException as e:
            logger.error("Failed to fetch V3 mints for batch starting at index %d after multiple retries: %s", i, e)
        except Exception as e:
            logger.error("A non-network error occurred during V3 mint fetching: %s", e)
            
    # Combine and format the results 
    processed_mints = {}
    for pool_id, mint_data in first_mints.items():
        pool_info = pool_info_map.get(pool_id)
        if pool_info:
            processed_mints[pool_id] = {
                "timestamp": int(mint_data["timestamp"]),
                "amount0": mint_data["amount0"],
                "amount1": mint_data["amount1"],
                "token0_address": pool_info.token0_address,
                "token1_address": pool_info.token1_address,
            }
    return processed_mints
# Implementation for this function and graph_fetch_token_reserves are similar, so we will fetch data for both in one go.
# @log_call
# def graph_fetch_first_mint_data(all_pools: Sequence[PoolInfo]) -> MintDataMap:
#     """
#     Fetch first LP (mint) data per pool from The Graph.

#     Should capture time from listing to first LP action in minutes.
#     """
#     # TODO: Implement query for first mint events per pool maybe with separate helper function for V2/V3.
#     raise NotImplementedError


@log_call
def call_fetch_deployers(all_pools: Sequence[PoolInfo]) -> Dict[str, str]:
    """
    Fetch pool deployer addresses via RPC (e.g., by inspecting transaction / logs).

    Returns {pool_address -> deployer_address}.
    """
    # TODO: Implement multicall / RPC logic to resolve deployers for each pool.
    """[STUB] Returns an empty map. To be implemented."""
    logger.warning("call_fetch_deployers is not implemented. Returning empty data.")
    return {}


@log_call
def call_fetch_current_owners(token_addresses: Sequence[str]) -> OwnerMap:
    """
    Fetch current owner for each token contract via RPC.

    Returns {token_address -> owner_address or None}.
    """
    # TODO: Implement owner() calls (possibly via multicall) for token contracts.
    """[STUB] Returns an empty map. To be implemented."""
    logger.warning("call_fetch_current_owners is not implemented. Returning empty data.")
    return {}


@log_call
def etherscan_get_token_sourcecode(
    token_addresses: Sequence[str],
) -> SourceCodeMap:
    """
    Fetch source code metadata (is_proxy, is_verified) for each token from Etherscan.
    """
    # TODO: Implement Etherscan getsourcecode endpoint usage.
    """[STUB] Returns an empty map. To be implemented."""
    logger.warning("etherscan_get_token_sourcecode is not implemented. Returning empty data.")
    return {}

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
    """[STUB] Returns an empty map. To be implemented."""
    logger.warning("call_fetch_code is not implemented. Returning empty data.")
    return {}


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
    """[STUB] Returns an empty map. To be implemented."""
    logger.warning("dune_fetch_data is not implemented. Returning empty data.")
    return {}


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
    """[STUB] Returns an empty map. To be implemented."""
    logger.warning("bitquery_fetch_data is not implemented. Returning empty data.")
    return {}


@log_call
def uniswap_web_fetch_data() -> Set[str]:
    """
    Fetch the set of verified tokens from https://tokens.uniswap.org/.

    Returns a set of token addresses that are considered trustworthy/verified.
    """
    # TODO: Implement HTTP fetch and JSON parsing of Uniswap token list.
    """[STUB] Returns an empty set. To be implemented."""
    logger.warning("uniswap_web_fetch_data is not implemented. Returning empty data.")
    return set()


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
    """
    t0_info = tokens.get(token0_address)
    t1_info = tokens.get(token1_address)

    is_t0_scam = t0_info.label if t0_info else None
    is_t1_scam = t1_info.label if t1_info else None

    # 1 if either token is explicitly labeled a scam
    if is_t0_scam or is_t1_scam:
        return 1.0

    # 0 if both tokens are trustworthy OR verified by Uniswap
    is_t0_trustworthy = (is_t0_scam is False) or (token0_address in uniswap_verified_tokens)
    is_t1_trustworthy = (is_t1_scam is False) or (token1_address in uniswap_verified_tokens)

    if is_t0_trustworthy and is_t1_trustworthy:
        return 0.0

    # If one is trustworthy and the other is unknown, use the global scam rate as a soft label
    if (is_t0_trustworthy and is_t1_scam is None) or \
       (is_t1_trustworthy and is_t0_scam is None):
        return scam_rate

    # Default for all other cases (e.g., both unknown)
    return scam_rate


@log_call
def build_unique_token_addresses(all_pools: Sequence[PoolInfo]) -> List[str]:
    """
    Extract a deduplicated list of token addresses from the full pool list.
    """
    unique_addresses = set()
    for pool in all_pools:
        unique_addresses.add(pool.token0_address)
        unique_addresses.add(pool.token1_address)
    return list(unique_addresses)   


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
    pool_counts: Dict[str, int],
) -> PoolMetricsRow:
    """
    Assemble a fully-populated PoolMetricsRow for a single pool using all
    pre-fetched data maps, gracefully handling missing data with None.
    """
    # Safely get data from maps, defaulting to None if the key doesn't exist
    pool_reserves = reserves_by_pool.get(pool.pool_address)
    token0_creation = creations.get(pool.token0_address)
    token1_creation = creations.get(pool.token1_address)
    token0_onchain = token_data.get(pool.token0_address)
    token1_onchain = token_data.get(pool.token1_address)
    first_mint_info = mint_data.get(pool.pool_address)
    token0_source = sourcecode_responses.get(pool.token0_address)
    token1_source = sourcecode_responses.get(pool.token1_address)
    pool_deployer_addr = pool_deployers.get(pool.pool_address)
    pool_deployer_dune = dune_data.get(pool_deployer_addr) if pool_deployer_addr else None
    token0_bitquery = bitquery_data.get(pool.token0_address)
    token1_bitquery = bitquery_data.get(pool.token1_address)
    
    # Helper to safely look up Dune metrics, which requires multiple dictionary lookups
    def get_dune_metric(token_addr: str, metric: str) -> Optional[int | float]:
        owner_addr = token_owners.get(token_addr)
        if not owner_addr: return None
        activity = dune_data.get(owner_addr)
        if not activity: return None
        return getattr(activity, metric, None)

    return PoolMetricsRow(
        pool_address=pool.pool_address,
        dex_name=dex_name,
        chain_name=chain_name,
        chain_id=chain_id,
        first_pool_activity_time=first_mint_info.first_pool_activity_time if first_mint_info else None,
        token0_address=pool.token0_address,
        token1_address=pool.token1_address,
        token0_name=pool.token0_name,
        token1_name=pool.token1_name,
        token0_symbol=pool.token0_symbol,
        token1_symbol=pool.token1_symbol,
        token0_pool_amount=pool_counts.get(pool.token0_address),
        token1_pool_amount=pool_counts.get(pool.token1_address),
        token0_total_supply=token0_onchain.total_supply if token0_onchain else None,
        token1_total_supply=token1_onchain.total_supply if token1_onchain else None,
        token0_age_minutes=int((pool.pool_creation_timestamp - token0_creation.creation_timestamp) / 60) if token0_creation else None,
        token1_age_minutes=int((pool.pool_creation_timestamp - token1_creation.creation_timestamp) / 60) if token1_creation else None,
        token0_verified_contract=token0_source.is_verified if token0_source else False,
        token1_verified_contract=token1_source.is_verified if token1_source else False,
        token0_proxy=token0_source.is_proxy if token0_source else False,
        token1_proxy=token1_source.is_proxy if token1_source else False,
        # Defaulting boolean values to False instead of None
        token0_ownership_renounced=token_owners.get(pool.token0_address) is None,
        token1_ownership_renounced=token_owners.get(pool.token1_address) is None,
        token0_owner_tx_count=get_dune_metric(pool.token0_address, 'tx_count'),
        token1_owner_tx_count=get_dune_metric(pool.token1_address, 'tx_count'),
        pool_deployer_tx_count=pool_deployer_dune.tx_count if pool_deployer_dune else None,
        token0_owner_age=get_dune_metric(pool.token0_address, 'address_age'),
        token1_owner_age=get_dune_metric(pool.token1_address, 'address_age'),
        pool_deployer_age=pool_deployer_dune.address_age if pool_deployer_dune else None,
        token0_owner_gas_burnt=get_dune_metric(pool.token0_address, 'gas_burnt'),
        token1_owner_gas_burnt=get_dune_metric(pool.token1_address, 'gas_burnt'),
        pool_deployer_gas_burnt=pool_deployer_dune.gas_burnt if pool_deployer_dune else None,
        token0_owner_bytes_deployed=get_dune_metric(pool.token0_address, 'bytes_deployed'),
        token1_owner_bytes_deployed=get_dune_metric(pool.token1_address, 'bytes_deployed'),
        pool_deployer_bytes_deployed=pool_deployer_dune.bytes_deployed if pool_deployer_dune else None,
        token0_owner_smart_contracts_interacted=get_dune_metric(pool.token0_address, 'smart_contracts_interacted'),
        token1_owner_smart_contracts_interacted=get_dune_metric(pool.token1_address, 'smart_contracts_interacted'),
        pool_deployer_smart_contracts_interacted=pool_deployer_dune.smart_contracts_interacted if pool_deployer_dune else None,
        token0_num_holders=token0_bitquery.num_holders if token0_bitquery else None,
        token1_num_holders=token1_bitquery.num_holders if token1_bitquery else None,
        token0_top_10_percent=token0_bitquery.top_10_percent if token0_bitquery else None,
        token1_top_10_percent=token1_bitquery.top_10_percent if token1_bitquery else None,
        token0_liquidity_depth=(pool_reserves.get(pool.token0_address) / token0_onchain.total_supply * 100) if pool_reserves and token0_onchain and token0_onchain.total_supply else None,
        token1_liquidity_depth=(pool_reserves.get(pool.token1_address) / token1_onchain.total_supply * 100) if pool_reserves and token1_onchain and token1_onchain.total_supply else None,
        label=calculate_label(pool.token0_address, pool.token1_address, tokens, scam_rate, uniswap_verified_tokens)
    )


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
    cache_dir = Path("output")


    v2_pairs_by_token = _load_or_fetch_data(
        cache_dir=cache_dir,
        base_filename="uniswap_v2_pairs",
        fetch_function=graph_batch_list_v2_pairs_for_tokens,
        reconstruct_class=PoolInfo, # <-- Tell it to reconstruct PoolInfo objects
        tokens=tokens,              
    )
    
    v3_pools_by_token = _load_or_fetch_data(
        cache_dir=cache_dir,
        base_filename="uniswap_v3_pools",
        fetch_function=graph_batch_list_v3_pools_for_tokens,
        reconstruct_class=PoolInfo, # <-- Tell it to reconstruct PoolInfo objects
        tokens=tokens,
    )
    logger.info("Loaded %d V2 pools and %d V3 pools.",
        sum(len(p) for p in v2_pairs_by_token.values()),
        sum(len(p) for p in v3_pools_by_token.values()),
    )

    # Build the global list of pools for processing
    all_pools: List[PoolInfo] = []
    for t_addr, pools in v2_pairs_by_token.items():
        all_pools.extend(pools)
    for t_addr, pools in v3_pools_by_token.items():
        all_pools.extend(pools)

    pool_counts = defaultdict(int)
    for pool in all_pools:
        pool_counts[pool.token0_address] += 1
        pool_counts[pool.token1_address] += 1
    





    # Build a deduplicated set of unique token addresses from all pools
    unique_token_addresses = build_unique_token_addresses(all_pools)

    # Fetch creation timestamp and deployer for each unique token address
    creations = etherscan_get_contracts_creation(unique_token_addresses)

    # Fetch token data (name, symbol, totalSupply, poolAmount, etc.)
    token_data = graph_fetch_token_data(unique_token_addresses, creations)

    # Fetch first LP data per pool
    first_mints_data = _load_or_fetch_data(
        cache_dir=cache_dir,
        base_filename="first_mints_raw_data",  
        fetch_function=_fetch_first_mints_for_pools,
        all_pools=all_pools,
    )
    
    mint_data = _process_mints_for_mint_data_map(
        first_mints_data, all_pools, save_to_dir=cache_dir
    )

    reserves_by_pool = _process_mints_for_reserves_by_pool(
        first_mints_data, save_to_dir=cache_dir
    )
    logger.info("Successfully processed data for %d mints and %d reserve pools.", len(mint_data), len(reserves_by_pool))

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
            pool_counts=pool_counts
        )
        results.append(metrics_row)

    return results



@log_call
def write_metrics_to_csv(rows: Sequence[PoolMetricsRow], output_path: Path) -> None:
    """
    Serialize the final list of PoolMetricsRow objects into a CSV file.
    """
    if not rows:
        logger.warning("No data rows to write to CSV.")
        return

    try:
        # We need to convert the list of dataclass objects into a list of dictionaries.
        # dataclasses.asdict is perfect for this.
        rows_as_dicts = [asdict(row) for row in rows]
        
        # The fieldnames for the CSV header will be the keys of the first dictionary.
        fieldnames = rows_as_dicts[0].keys()

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header row
            writer.writeheader()
            
            # Write all the data rows
            writer.writerows(rows_as_dicts)
        
        logger.info("Successfully wrote %d rows to %s", len(rows), output_path)

    except (IOError, csv.Error) as e:
        logger.exception("Failed to write metrics to CSV file %s: %s", output_path, e)
        raise


@log_call
def main() -> None:
    """
    CLI entrypoint: load input CSV, compute metrics, and write them to disk.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Collect static pool metrics for a categorized token list.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("categorized_tokens.csv"),
        help="Path to categorized_tokens.csv file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("static_pool_metrics.csv"),
        help="Path where the output metrics CSV will be written.",
    )
    args = parser.parse_args()
    # function below is executed for testing with thegraph list only

    logger.info("Loading tokens from %s", args.input_csv)
    rows = calculate_metrics_for_token_list(args.input_csv)

    logger.info("Writing %d pool metrics rows to %s", len(rows), args.output_csv)
    write_metrics_to_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()

