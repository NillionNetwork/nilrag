"""The SecretVault organization configuration"""

import os

from dotenv import load_dotenv

load_dotenv()

# Organization configuration
ORG_CONFIG = {
    "org_credentials": {
        "secret_key": os.getenv("NILLION_ORG_SECRET_KEY"),
        "org_did": os.getenv("NILLION_ORG_DID"),
    },
    "nodes": [
        {
            "url": os.getenv("URL1"),
            "did": os.getenv("DID1"),
        },
        {
            "url": os.getenv("URL2"),
            "did": os.getenv("DID2"),
        },
        {
            "url": os.getenv("URL3"),
            "did": os.getenv("DID3"),
        },
    ],
}
