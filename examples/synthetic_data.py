import argparse
from faker import Faker

def generate_fake_profiles(rows: int, output_file: str):
    fake = Faker()

    with open(output_file, "w", encoding="utf-8") as file:
        for _ in range(rows):
            name = fake.name()
            sentence = (
                f"{name} works at {fake.company()} as a {fake.job()}. "
                f"{name} was born on {fake.profile()['birthdate']} "
                f"and lives at {fake.address().replace('\n', ', ')}."
            )
            file.write(sentence + "\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake profiles.")
    parser.add_argument("--rows", type=int, default=10, help="Number of fake profiles to generate.")
    parser.add_argument("--output", type=str, default="fake_profiles.txt", help="Output file name.")
    args = parser.parse_args()

    generate_fake_profiles(args.rows, args.output)
