import argparse
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Call HAM10000 BentoML service")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--url",
        default="http://localhost:3000/predict",
        help="BentoML predict endpoint",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions to return",
    )
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        files = {"image": f}
        response = requests.post(
            args.url,
            params={"top_k": args.top_k},
            files=files,
            timeout=60,
        )

    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
