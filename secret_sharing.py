import random
import numpy as np

class Secret:
    def __init__(self, share, precision):
        self.share = share
        self.precision = precision

    def __repr__(self):
        return f"Secret(share={self.share}, precision={self.precision})"

# Additive Secret Sharing
class AdditiveSecretSharing:
    # Initialize the secret sharing system.
    def __init__(self, num_parties, prime_mod, precision=4):
        self.num_parties = num_parties
        self.prime_mod = prime_mod
        self.precision = precision
        self.scaling_factor = 10 ** precision

    # Create shares for a single secret.
    def _secret_share(self, secret):
        is_float = isinstance(secret, (float, np.float32))
        secret = self._to_fixed_point(secret) if is_float else secret
        shares = [
            Secret(
                random.randint(0, self.prime_mod - 1),
                self.precision if is_float else 0
            ) for _ in range(self.num_parties - 1)
        ]
        final_share = Secret(
            (secret - sum([s.share for s in shares])) % self.prime_mod,
            self.precision if is_float else 0
        )
        shares.append(final_share)
        return shares

    # Secret share a value or list of values.
    def secret_share(self, secret_or_list):
        if isinstance(secret_or_list, (int, float, np.float32)):
            is_float = isinstance(secret_or_list, float) or isinstance(secret_or_list, np.float32)
        elif isinstance(secret_or_list, list) or isinstance(secret_or_list, np.ndarray):
            is_float = all(isinstance(x, float) or isinstance(x, np.float32) for x in secret_or_list)
            is_int = all(isinstance(x, int) for x in secret_or_list)
            if not (is_float or is_int):
                raise TypeError("All elements in the list must be either integers or floats.")
        else:
            raise TypeError(f"Input must be an integer, float, or a list of such values. Got {type(secret_or_list)}.")

        if isinstance(secret_or_list, (int, float, np.float32)):  # Single secret
            return self._secret_share(secret_or_list)
        elif isinstance(secret_or_list, list) or isinstance(secret_or_list, np.ndarray):  # List of secrets
            all_shares = [[] for _ in range(self.num_parties)]
            for secret in secret_or_list:
                shares = self._secret_share(secret)
                for i in range(self.num_parties):
                    all_shares[i].append(shares[i])
            return all_shares

    # Reconstruct a secret or list of secrets from shares.
    def secret_reveal(self, shares_or_list):
        if all(isinstance(share, Secret) for share in shares_or_list):  # Single secret
            modified_secret = sum([s.share for s in shares_or_list]) % self.prime_mod
            share_precision = shares_or_list[0].precision
            assert all(share_precision == share.precision for share in shares_or_list)
            return self._from_fixed_point(modified_secret) if share_precision > 0 else modified_secret
        elif all(isinstance(share, list) or isinstance(share, np.ndarray) for share in shares_or_list):  # List of secrets
            secrets = []
            for i in range(len(shares_or_list[0])):
                modified_secret = sum(shares_or_list[party][i].share for party in range(self.num_parties)) % self.prime_mod
                share_precision = shares_or_list[0][i].precision
                assert all(share_precision == shares_or_list[party][i].precision for party in range(self.num_parties))
                secrets.append(
                    self._from_fixed_point(modified_secret) if share_precision > 0 else modified_secret
                )
            return secrets
        else:
            raise TypeError("Input must be a list of integers or a list of lists of integers.")

    # Convert a floating-point value to fixed-point.
    def _to_fixed_point(self, value):
        return int(round(value * self.scaling_factor))

    # Convert a fixed-point value back to floating-point.
    def _from_fixed_point(self, value):
        return value / self.scaling_factor

def secret_sharing_test():
    secret_sharing = AdditiveSecretSharing(3, 2**31 - 1, 5)

    # Single int example
    secret = 100
    shares = secret_sharing.secret_share(secret)
    print(f"Shares: {shares}")
    reconstructed_secret = secret_sharing.secret_reveal(shares)
    print(f"Reconstructed Secret: {reconstructed_secret}")
    assert secret == reconstructed_secret

    # Single float example
    secret_float = 123.45
    shares_float = secret_sharing.secret_share(secret_float)
    print(f"Shares (single float): {shares_float}")
    reconstructed_float = secret_sharing.secret_reveal(shares_float)
    print(f"Reconstructed secret (single float): {reconstructed_float}")
    assert secret_float == reconstructed_float

    # Float list example
    secret_list = [123.0, 456.78, 789.0]
    shares_list = secret_sharing.secret_share(secret_list)
    print(f"Shares (list of secrets):\n{shares_list}")
    reconstructed_list = secret_sharing.secret_reveal(shares_list)
    print(f"Reconstructed secrets (list): {reconstructed_list}")
    assert [secret == reconstructed for secret, reconstructed in zip(secret_list, reconstructed_list)]

