"""
Transform util functions
"""

import nilql

PRECISION = 7
SCALING_FACTOR = 10**PRECISION


def group_shares_by_id(shares_per_party: list, transform_share_fn: callable):
    """
    Groups shares by their ID and applies a transform function to each share.

    Args:
        shares_per_party (list): List of shares from each party
        transform_share_fn (callable): Function to transform each share value

    Returns:
        dict: Dictionary mapping IDs to list of transformed shares
    """
    shares_by_id = {}
    for party_shares in shares_per_party:
        for share in party_shares["shares"]:
            share_id = share["_id"]
            if share_id not in shares_by_id:
                shares_by_id[share_id] = []
            shares_by_id[share_id].append(transform_share_fn(share))
    return shares_by_id


def to_fixed_point(value: float) -> int:
    """
    Convert a floating-point value to fixed-point representation.

    Args:
        value (float): Value to convert

    Returns:
        int: Fixed-point representation with PRECISION decimal places
    """
    return int(round(value * SCALING_FACTOR))


def from_fixed_point(value: int) -> float:
    """
    Convert a fixed-point value back to a floating-point.

    Args:
        value (int): Fixed-point value to convert

    Returns:
        float: Floating-point representation
    """
    return value / SCALING_FACTOR


def encrypt_float_list(sk, lst: list[float]) -> list[list]:
    """
    Encrypt a list of floats using a secret key.

    Args:
        sk: Secret key for encryption
        lst (list): List of float values to encrypt

    Returns:
        list: List of encrypted fixed-point values
    """
    return [nilql.encrypt(sk, to_fixed_point(l)) for l in lst]


def decrypt_float_list(sk, lst: list[list]) -> list[float]:
    """
    Decrypt a list of encrypted fixed-point values to floats.

    Args:
        sk: Secret key for decryption
        lst (list): List of encrypted fixed-point values

    Returns:
        list: List of decrypted float values
    """
    return [from_fixed_point(nilql.decrypt(sk, l)) for l in lst]


def encrypt_string_list(sk, lst: list) -> list:
    """
    Encrypt a list of strings using a secret key.

    Args:
        sk: Secret key for encryption
        lst (list): List of strings to encrypt

    Returns:
        list: List of encrypted strings
    """
    return [nilql.encrypt(sk, l) for l in lst]


def decrypt_string_list(sk, lst: list) -> list:
    """
    Decrypt a list of encrypted strings.

    Args:
        sk: Secret key for decryption
        lst (list): List of encrypted strings

    Returns:
        list: List of decrypted strings
    """
    return [nilql.decrypt(sk, l) for l in lst]
