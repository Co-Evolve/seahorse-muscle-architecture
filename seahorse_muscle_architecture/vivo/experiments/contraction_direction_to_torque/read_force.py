import time
from datetime import datetime

import pandas as pd
import serial

if __name__ == '__main__':
    SERIAL_PORT = '/dev/tty.usbmodemFFFFFFFEFFFF1'
    BAUDRATE = 9600

    output_path = f"data/{datetime.now()}_FORCE_LOG.csv"
    df = pd.DataFrame(columns=["time", "force"])
    ser = serial.Serial(SERIAL_PORT, BAUDRATE)

    try:
        while True:
            line = str(ser.readline().decode("utf-8"))
            print(line)
            force = float(line[line.find("Force:") + len("Force:"):line.find(" N")])

            df.loc[len(df.index)] = [time.time(), force]
    except KeyboardInterrupt:
        df.to_csv(path_or_buf=output_path, index=False, sep="\t")
