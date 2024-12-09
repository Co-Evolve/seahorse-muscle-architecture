from __future__ import annotations

import logging
import sys
import time
from typing import List, Optional

import numpy as np
from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


class Controller:
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_GOAL_CURRENT = 102
    ADDR_PRESENT_POSITION = 132
    ADDR_PRESENT_CURRENT = 126
    ADDR_OPERATING_MODE = 11
    ADDR_P_GAIN = 84
    DXL_MINIMUM_POSITION_VALUE = 0  # Refer to the CW Angle Limit of product eManual
    DXL_MAXIMUM_POSITION_VALUE = 4095  # Refer to the CCW Angle Limit of product eManual
    DXL_MINIMUM_CURRENT_VALUE = -2047
    DXL_MAXIMUM_CURRENT_VALUE = 2047
    BAUDRATE = 1000000
    PROTOCOL_VERSION = 2.0
    DEVICENAME = '/dev/tty.usbserial-FT7928YF'
    POSITION_ERROR_THRESHOLD = 5  # Dynamixel moving status threshold

    def __init__(
            self,
            dxl_ids: List[int],
            use_degrees: bool
            ) -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Controller")

        if use_degrees:
            self.POSITION_ERROR_THRESHOLD = 1

        self.dxl_ids = dxl_ids
        self._use_degrees = use_degrees
        self._port_handler = self._setup_port_handler()
        self._set_port_handler_baudrate()
        self._packet_handler = self._setup_packet_handler()
        self._dxl_id_to_current_goal_position = dict()

    def _setup_port_handler(
            self
            ) -> PortHandler:
        port_handler = PortHandler(port_name=self.DEVICENAME)
        if port_handler.openPort():
            self.logger.info(f"Succeeded to open port: {self.DEVICENAME}")
        else:
            self.logger.error(f"Failed to open port: {self.DEVICENAME}")
            sys.exit(1)
        return port_handler

    def _set_port_handler_baudrate(
            self
            ) -> None:
        if self._port_handler.setBaudRate(baudrate=self.BAUDRATE):
            self.logger.info(f"Succeeded to change the PortHandler's baudrate: {self.BAUDRATE}")
        else:
            self.logger.error(f"Failed to change the PortHandler's baudrate: {self.BAUDRATE}")
            sys.exit(1)

    def _setup_packet_handler(
            self, ) -> PacketHandler:
        packet_handler = PacketHandler(protocol_version=self.PROTOCOL_VERSION)
        return packet_handler

    def set_operating_mode(
            self,
            dxl_id: int,
            mode: str
            ) -> None:
        if mode == "position":
            data = 3
        elif mode == "current":
            data = 0
        else:
            raise ValueError

        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
                port=self._port_handler, dxl_id=dxl_id, address=self.ADDR_OPERATING_MODE, data=data
                )
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getRxPacketError(dxl_error)}")
        else:
            self.logger.info(f"[ID: {dxl_id}] Set operating mode: {mode} control")

    def _position_to_degrees(
            self,
            position: int
            ) -> float:
        return 360 * (position / self.DXL_MAXIMUM_POSITION_VALUE)

    def _degrees_to_position(
            self,
            degrees: float
            ) -> int:
        return round(self.DXL_MAXIMUM_POSITION_VALUE * (degrees / 360))

    def torque_enabled(
            self,
            dxl_id: int,
            mode: Optional[int] = 1
            ) -> None:
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
                port=self._port_handler, dxl_id=dxl_id, address=self.ADDR_TORQUE_ENABLE, data=mode
                )
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getRxPacketError(dxl_error)}")
        else:
            self.logger.info(f"[ID: {dxl_id}] Dynamixel torque enabled: {mode}")

    def set_goal_position(
            self,
            dxl_id: int,
            goal_position: int
            ) -> None:
        if self._use_degrees:
            goal_position = self._degrees_to_position(degrees=goal_position)
        goal_position = np.clip(goal_position, self.DXL_MINIMUM_POSITION_VALUE, self.DXL_MAXIMUM_POSITION_VALUE)
        dxl_comm_result, dxl_error = self._packet_handler.write4ByteTxRx(
                port=self._port_handler, dxl_id=dxl_id, address=self.ADDR_GOAL_POSITION, data=goal_position
                )
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getRxPacketError(dxl_error)}")
        else:
            if self._use_degrees:
                goal_position = self._position_to_degrees(position=goal_position)
            self.logger.info(f"[ID: {dxl_id}] Goal position set to {goal_position}")
            self._dxl_id_to_current_goal_position[dxl_id] = goal_position

    def set_goal_current(
            self,
            dxl_id: int,
            goal_current: int
            ) -> None:
        goal_current = np.clip(goal_current, self.DXL_MINIMUM_CURRENT_VALUE, self.DXL_MAXIMUM_CURRENT_VALUE)
        dxl_comm_result, dxl_error = self._packet_handler.write2ByteTxRx(
                port=self._port_handler, dxl_id=dxl_id, address=self.ADDR_GOAL_CURRENT, data=goal_current
                )
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getRxPacketError(dxl_error)}")
        else:
            self.logger.info(f"[ID: {dxl_id}] Goal current set to {goal_current}")

    def get_position(
            self,
            dxl_id: int
            ) -> Optional[int]:
        while True:
            try:
                dxl_present_position, dxl_comm_result, dxl_error = self._packet_handler.read4ByteTxRx(
                        port=self._port_handler, dxl_id=dxl_id, address=self.ADDR_PRESENT_POSITION
                        )
                if dxl_comm_result != COMM_SUCCESS:
                    self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getTxRxResult(dxl_comm_result)}")
                elif dxl_error != 0:
                    self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getRxPacketError(dxl_error)}")
                else:
                    if self._use_degrees:
                        dxl_present_position = self._position_to_degrees(position=dxl_present_position)
                    self.logger.info(f"[ID: {dxl_id}] Motor position: {dxl_present_position:3f}")
                    return dxl_present_position
                return
            except IndexError:
                pass

    def get_current(
            self,
            dxl_id: int
            ) -> Optional[int]:
        while True:
            try:
                dxl_present_current, dxl_comm_result, dxl_error = self._packet_handler.read2ByteTxRx(
                        port=self._port_handler, dxl_id=dxl_id, address=self.ADDR_PRESENT_CURRENT
                        )
                if dxl_comm_result != COMM_SUCCESS:
                    self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getTxRxResult(dxl_comm_result)}")
                elif dxl_error != 0:
                    self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getRxPacketError(dxl_error)}")
                else:
                    self.logger.info(f"[ID: {dxl_id}] Motor current: {dxl_present_current:3f}")
                    return dxl_present_current
                return
            except IndexError:
                pass

    def set_p_gain(
            self,
            dxl_id: int,
            gain: int = 850
            ) -> None:
        gain = np.clip(gain, 0, 850)
        dxl_comm_result, dxl_error = self._packet_handler.write2ByteTxRx(
                port=self._port_handler, dxl_id=dxl_id, address=self.ADDR_P_GAIN, data=gain
                )
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            self.logger.error(f"[ID: {dxl_id}] {self._packet_handler.getRxPacketError(dxl_error)}")
        else:
            self.logger.info(f"[ID: {dxl_id}] Position control gain set to {gain}")

    def reset_all_p_gains(
            self
            ) -> None:
        for dxl_id in self.dxl_ids:
            self.set_p_gain(dxl_id=dxl_id)

    def wait_till_position_reached(
            self,
            dxl_id: int,
            timeout: float = 5
            ) -> None:
        start_time = time.time()
        while dxl_id in self._dxl_id_to_current_goal_position and (time.time() - start_time < timeout):
            goal_position = self._dxl_id_to_current_goal_position[dxl_id]
            current_position = self.get_position(dxl_id=dxl_id)
            error = abs(goal_position - current_position) if current_position is not None else np.inf
            if error < self.POSITION_ERROR_THRESHOLD:
                self._dxl_id_to_current_goal_position.pop(dxl_id)
            time.sleep(0.1)

    def wait_till_all_positions_reached(
            self,
            timeout: float = 5
            ) -> None:
        start_time = time.time()
        while self._dxl_id_to_current_goal_position and (time.time() - start_time < timeout):
            done_dxl_ids = []
            for dxl_id in self._dxl_id_to_current_goal_position:
                goal_position = self._dxl_id_to_current_goal_position[dxl_id]
                current_position = self.get_position(dxl_id=dxl_id)
                error = abs(goal_position - current_position) if current_position is not None else np.inf
                if error < self.POSITION_ERROR_THRESHOLD:
                    done_dxl_ids.append(dxl_id)

            for dxl_id in done_dxl_ids:
                self._dxl_id_to_current_goal_position.pop(dxl_id)
            time.sleep(0.1)

    def cleanup(
            self
            ) -> None:
        if self._port_handler.is_open:
            self.reset_all_p_gains()
            for dxl_id in self.dxl_ids:
                self.torque_enabled(dxl_id=dxl_id, mode=0)
            self._port_handler.closePort()


class MX106MotorController:
    def __init__(
            self,
            dxl_ids: List[int],
            use_degrees: bool
            ) -> None:
        self.dxl_ids = dxl_ids
        self.use_degrees = use_degrees

    def __enter__(
            self, ) -> Controller:
        self._controller = Controller(dxl_ids=self.dxl_ids, use_degrees=self.use_degrees)
        return self._controller

    def __exit__(
            self,
            exc_type,
            exc_val,
            exc_tb
            ) -> None:
        self._controller.cleanup()

    def __del__(
            self
            ) -> None:
        self._controller.cleanup()
