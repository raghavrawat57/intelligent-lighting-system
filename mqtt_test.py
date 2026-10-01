import json
import random
import time
from datetime import datetime

import paho.mqtt.client as mqtt


BROKER = "localhost"
PORT = 1883
TOPIC = "smartlighting/sensors"


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.connect(BROKER, PORT, 60)
client.loop_start()

print("ESP32 simulator started...")
print("Sending sensor data every second.\n")


try:

    while True:

        now = datetime.now()

        hour = now.hour

        # -----------------------------------
        # OCCUPANCY
        # -----------------------------------

        if 8 <= hour < 18:
            occupancy = random.randint(1, 8)

        elif 18 <= hour < 22:
            occupancy = random.randint(2, 10)

        else:
            occupancy = random.randint(0, 3)


        # -----------------------------------
        # AMBIENT LIGHT
        # -----------------------------------

        if 8 <= hour < 17:

            ambient_light = random.randint(400, 900)

        elif 17 <= hour < 20:

            ambient_light = random.randint(100, 400)

        else:

            ambient_light = random.randint(5, 100)


        # -----------------------------------
        # TEMPERATURE
        # -----------------------------------

        temperature = round(
            random.uniform(23, 32),
            1
        )


        # -----------------------------------
        # LIGHTING DEMAND
        # -----------------------------------

        # More people → more light
        # Less ambient light → more artificial light

        demand = (
            occupancy * 7
            + (900 - ambient_light) / 12
        )

        demand = max(0, min(100, demand))

        demand = round(demand)


        # -----------------------------------
        # INDIVIDUAL LIGHTS
        # -----------------------------------

        light_1 = max(
            0,
            min(100, demand + random.randint(-5, 5))
        )

        light_2 = max(
            0,
            min(100, demand + random.randint(-8, 8))
        )

        light_3 = max(
            0,
            min(100, demand + random.randint(-5, 5))
        )

        light_4 = max(
            0,
            min(100, demand + random.randint(-8, 8))
        )


        # -----------------------------------
        # DATA PACKET
        # -----------------------------------

        data = {

            "timestamp":
                now.strftime("%Y-%m-%d %H:%M:%S"),

            "occupancy":
                occupancy,

            "ambient_light":
                ambient_light,

            "temperature":
                temperature,

            "lighting_demand":
                demand,

            "light_1":
                light_1,

            "light_2":
                light_2,

            "light_3":
                light_3,

            "light_4":
                light_4
        }


        message = json.dumps(data)


        # -----------------------------------
        # MQTT PUBLISH
        # -----------------------------------

        client.publish(
            TOPIC,
            message,
            qos=1
        )


        print(
            f"{data['timestamp']} | "
            f"People: {occupancy} | "
            f"Light: {ambient_light} lux | "
            f"Temp: {temperature}°C | "
            f"Demand: {demand}%"
        )


        time.sleep(1)


except KeyboardInterrupt:

    print("\nSimulator stopped.")


finally:

    client.loop_stop()
    client.disconnect()