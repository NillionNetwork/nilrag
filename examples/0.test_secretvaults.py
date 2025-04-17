from secretvaults import *

"""The SecretVault organization configuration"""

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

# Organization configuration
org_config = {
    "org_credentials": {
        "secret_key": os.getenv("NILLION_ORG_SECRET_KEY"),
        "org_did": os.getenv("NILLION_ORG_DID"),
    },
    "nodes": [
        {
            "url": "https://nildb-nx8v.nillion.network",
            "did": "did:nil:testnet:nillion1qfrl8nje3nvwh6cryj63mz2y6gsdptvn07nx8v",
        },
        {
            "url": "https://nildb-p3mx.nillion.network",
            "did": "did:nil:testnet:nillion1uak7fgsp69kzfhdd6lfqv69fnzh3lprg2mp3mx",
        },
        {
            "url": "https://nildb-rugk.nillion.network",
            "did": "did:nil:testnet:nillion1kfremrp2mryxrynx66etjl8s7wazxc3rssrugk",
        },
    ],
}


async def main():
    """
    Main function to print the org config, initialize the SecretVaultWrapper,
    and generate API tokens for all nodes.
    """
    try:
        # Initialize the SecretVaultWrapper instance with the org configuration
        org = SecretVaultWrapper(org_config["nodes"], org_config["org_credentials"])
        print(f"{org.nodes}")
        await org.init()

        # Generate API tokens for all nodes
        api_tokens = await org.generate_tokens_for_all_nodes()
        print("🪙 API Tokens:", json.dumps(api_tokens, indent=2))

    except RuntimeError as error:
        print(f"❌ Failed to use SecretVaultWrapper: {str(error)}")


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
