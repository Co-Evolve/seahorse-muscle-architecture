import time
from datetime import datetime

import pandas as pd

from seahorse_muscle_architecture.vivo.experiments.controller import MX106MotorController

if __name__ == '__main__':
    dxl_id = 3
    START_CURRENT = 3
    END_CURRENT = 10
    DURATION = 10

    output_path = f"data/{datetime.now()}_CURRENT_LOG.csv"
    df = pd.DataFrame(columns=["time", "present_current", "goal_current", "position_shift"])

    with MX106MotorController(dxl_ids=[dxl_id], use_degrees=True) as controller:
        controller.torque_enabled(dxl_id=dxl_id, mode=0)
        controller.set_operating_mode(dxl_id=dxl_id, mode="current")
        controller.torque_enabled(dxl_id=dxl_id, mode=1)
        controller.set_goal_current(dxl_id=dxl_id, goal_current=0)
        controller.get_current(dxl_id=dxl_id)
        original_position = controller.get_position(dxl_id=dxl_id)

        start_time = time.time()

        while time.time() - start_time < DURATION:
            alpha = (time.time() - start_time) / DURATION
            goal_current = round(START_CURRENT + alpha * (END_CURRENT - START_CURRENT))
            controller.set_goal_current(dxl_id=dxl_id, goal_current=goal_current)

            present_current = controller.get_current(dxl_id=dxl_id)
            present_position = controller.get_position(dxl_id=dxl_id)

            df.loc[len(df.index)] = [time.time(), present_current * 3.36, goal_current,
                                     present_position - original_position]

    df.to_csv(path_or_buf=output_path, index=False, sep="\t")
