# client_test.py
import time
import random
import requests


API_URL = "http://127.0.0.1:8000/api/score_transaction"


def build_sample_payload() -> dict:
    now = int(time.time())

    return {
        "cc_num": 4321987654321098,              # fixed card so it can build history
        "unix_time": now,
        "amt": round(random.uniform(10, 8000), 2),
        "merchant_id": None,
        "merchant_name": random.choice(
            ["AMAZON_ONLINE", "TEST_GROCERY", "LUXURY_ELECTRONICS_WEB"]
        ),
        "category": random.choice(["shopping_net", "grocery_pos", "misc_net"]),
        "lat": 12.9716,      # say Bangalore
        "lon": 77.5946,
        "merch_lat": 40.7128,  # say New York
        "merch_lon": -74.0060,
        "txn_datetime": None,
        "source_system": "api_client_test",
    }


def main():
    payload = build_sample_payload()
    print("Sending payload:")
    print(payload)

    resp = requests.post(API_URL, json=payload)
    print("\nStatus code:", resp.status_code)
    print("Response JSON:")
    print(resp.json())


if __name__ == "__main__":
    main()
