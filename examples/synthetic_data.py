"""
Script to generate synthetic data for testing nilRAG.

Usage:
uv run examples/synthetic_data.py --rows 100 --output examples/data/100-fake.txt
"""

import argparse

from faker import Faker


def generate_fake_profiles(rows: int, output_file: str):
    """
    Generate synthetic profiles and write them to a specified output file.

    Args:
        rows (int): The number of fake profiles to generate.
        output_file (str): The path to the output file where profiles will be saved.

    Each profile includes a name, job title, company, birthdate, and address.
    Profiles are separated by double newlines in the output file.
    """
    fake = Faker()

    with open(output_file, "w", encoding="utf-8") as file:
        for _ in range(rows):
            name = fake.name()
            address = fake.address().replace("\n", ", ")
            sentence = (
                f"{name} works at {fake.company()} as a {fake.job()}. "
                f"{name} was born on {fake.profile()['birthdate']} "
                f"and lives at {address}."
            )
            file.write(sentence + "\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake profiles.")
    parser.add_argument(
        "--rows", type=int, default=10, help="Number of fake profiles to generate."
    )
    parser.add_argument(
        "--output", type=str, default="fake_profiles.txt", help="Output file name."
    )
    args = parser.parse_args()

    generate_fake_profiles(args.rows, args.output)
