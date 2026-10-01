import json
import csv
import os
from datetime import datetime

import paho.mqtt.client as mqtt


BROKER = "localhost"
PORT = 1883
TOPIC = "smartlighting/sensors"

CSV_FILE = "sensor_data.csv"


def save_to_csv(data):

    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "occupancy",
                "ambient_light",
                "temperature",
                "lighting_demand",
                "light_1",
                "light_2",
                "light_3",
                "light_4"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data["occupancy"],
            data["ambient_light"],
            data["temperature"],
            data.get("lighting_demand", 0),
            data.get("light_1", 0),
            data.get("light_2", 0),
            data.get("light_3", 0),
            data.get("light_4", 0)
        ])

    print("✅ Saved to CSV")


def on_connect(client, userdata, flags, reason_code, properties):

    print("✅ Connected to Mosquitto")

    client.subscribe(TOPIC)

    print("📡 Listening:", TOPIC)


def on_message(client, userdata, message):

    try:

        payload = message.payload.decode()

        print("\n📥 MQTT received:")
        print(payload)

        data = json.loads(payload)

        save_to_csv(data)

    except Exception as e:

        print("❌ Error:", e)


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.on_connect = on_connect
client.on_message = on_message

print("Connecting to MQTT broker...")

client.connect(BROKER, PORT, 60)

client.loop_forever()