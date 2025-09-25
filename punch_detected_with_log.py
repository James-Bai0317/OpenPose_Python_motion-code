# 重構後的punch.py - 專注於動作辨識，按run後在Pycharm上打開攝影機，可先按一次q即可重製攝影機，測試完後再按q可退出攝影機
import math
import time
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
import csv # 紀錄log檔日誌，以便整理實驗數據

# 引入 OpenPose API
from pose_capture.openpose_api import get_keypoints_stream

# 引入angle.py或使用內建函數
try:
    from angle import calculate_normalized_angle, calculate_shoulder_width
    from angle import (NOSE, NECK, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
                       LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, MID_HIP,
                       RIGHT_HIP, LEFT_HIP, RIGHT_KNEE, LEFT_KNEE,
                       RIGHT_ANKLE, LEFT_ANKLE)

    print("Successfully imported angle module functions")
except ImportError:
    print("Warning: angle.py cannot find, using built-in angle functions")

    # 內建的關鍵點索引
    NOSE = 0
    NECK = 1
    RIGHT_SHOULDER = 2
    RIGHT_ELBOW = 3
    RIGHT_WRIST = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    MID_HIP = 8
    RIGHT_HIP = 9
    LEFT_HIP = 10
    RIGHT_KNEE = 11
    LEFT_KNEE = 12
    RIGHT_ANKLE = 13
    LEFT_ANKLE = 14


    def calculate_normalized_angle(keypoints, joint_indices, person_index):
        """計算標準化角度 - 內置版本"""
        try:
            if person_index >= len(keypoints):
                return None

            a_index, b_index, c_index = joint_indices
            person_keypoints = keypoints[person_index]

            if len(person_keypoints.shape) == 2:
                a = person_keypoints[a_index][:2] if person_keypoints[a_index][2] > 0.3 else [0, 0]
                b = person_keypoints[b_index][:2] if person_keypoints[b_index][2] > 0.3 else [0, 0]
                c = person_keypoints[c_index][:2] if person_keypoints[c_index][2] > 0.3 else [0, 0]
            else:
                return None

            if a[0] == 0 or a[1] == 0 or b[0] == 0 or b[1] == 0 or c[0] == 0 or c[1] == 0:
                return None

            shoulder_width = calculate_shoulder_width(keypoints, person_index)
            if shoulder_width is None or shoulder_width < 1e-10:
                return None

            ba_norm = [(a[0] - b[0]) / shoulder_width, (a[1] - b[1]) / shoulder_width]
            bc_norm = [(c[0] - b[0]) / shoulder_width, (c[1] - b[1]) / shoulder_width]

            dot_product = ba_norm[0] * bc_norm[0] + ba_norm[1] * bc_norm[1]
            mag_ba = math.sqrt(ba_norm[0] ** 2 + ba_norm[1] ** 2)
            mag_bc = math.sqrt(bc_norm[0] ** 2 + bc_norm[1] ** 2)

            if mag_ba < 1e-10 or mag_bc < 1e-10:
                return None

            cos_angle = dot_product / (mag_ba * mag_bc)
            cos_angle = max(min(cos_angle, 1.0), -1.0)
            angle_rad = math.acos(cos_angle)
            angle_deg = angle_rad * 180.0 / math.pi
            return angle_deg

        except (IndexError, TypeError, ValueError):
            return None


    def calculate_shoulder_width(keypoints, person_index):
        """計算肩寬作為身體比例標準 - 內置版本"""
        try:
            if person_index >= len(keypoints):
                return None

            person_keypoints = keypoints[person_index]

            if len(person_keypoints.shape) == 2:
                left_shoulder = person_keypoints[LEFT_SHOULDER][:2] if person_keypoints[LEFT_SHOULDER][2] > 0.3 else [0,
                                                                                                                      0]
                right_shoulder = person_keypoints[RIGHT_SHOULDER][:2] if person_keypoints[RIGHT_SHOULDER][
                                                                             2] > 0.3 else [0, 0]
            else:
                return None

            if left_shoulder[0] == 0 or right_shoulder[0] == 0:
                return None

            return math.sqrt((left_shoulder[0] - right_shoulder[0]) ** 2 + (left_shoulder[1] - right_shoulder[1]) ** 2)
        except (IndexError, TypeError):
            return None

# 引入guard_detection.py
from guard_detection import BoxingDefenseType, BoxingGuardConfig, PlayerDefenseData, DefenseFrameData, BoxingDefenseDetector, BoxingDefenseVisualizer

#以下定義參數資料型別

# 定義動作類型
class ActionType(Enum):
    IDLE = "idle"
    PUNCH_STRAIGHT = "straight_punch"
    PUNCH_HOOK = "hook_punch"
    PUNCH_UPPERCUT = "uppercut_punch"
    GUARD = "guard"
    DODGE = "dodge"

# 拳法配置的資料型別
@dataclass
class PunchConfig:
    name: str
    angle_threshold: float
    speed_threshold: float
    min_extension: float

# 統計玩家動作數據資料型別
@dataclass
class PlayerActionData:
    player_id: int
    action_type: str
    attack_hand: Optional[str]
    punch_type: Optional[str]
    velocity: float
    confidence: float
    arm_angles: Dict[str, Optional[float]]
    shoulder_width: Optional[float]
    body_center: Optional[Tuple[float, float]]
    is_attacking: bool
    is_guarding: bool
    timestamp: float

# 單幀數據
@dataclass
class FrameData:
    frame_id: int
    timestamp: float
    players: List[PlayerActionData]
    players_distance: Optional[float]


class BoxingActionDetector:
    """拳擊動作檢測器 - 純動作辨識，不含unity通訊邏輯"""

    def __init__(self):
        self.frame_count = 0

        # 拳法配置
        self.punch_configs = {
            "straight": PunchConfig("直拳", 120, 0.02, 70),
            "hook": PunchConfig("勾拳", 80, 0.03, 70),
            "uppercut": PunchConfig("上勾拳", 50, 0.02, 70)
        }

        # 防禦設置
        self.guard_angle_min = 40
        self.guard_angle_max = 140

        # 位置歷史記錄 (用於計算速度)
        self.position_history = {}

        # 玩家動作冷卻
        self.player_cooldowns = {}
        self.cooldown_frames = 10

        print("=== Boxing Action Detector Initialized ===")

        # 添加防禦檢測器
        self.defense_detector = BoxingDefenseDetector()
        print("Defense detector initialized")

    def detect_actions(self, keypoints) -> FrameData:
        """檢測所有玩家的動作"""
        self.frame_count += 1
        timestamp = time.time()

        players_data = []
        players_distance = None

        if keypoints is not None and len(keypoints) > 0:
            # 確保最多處理兩個人
            num_persons = min(len(keypoints), 2)

            for person_id in range(num_persons):
                player_data = self._detect_player_action(person_id, keypoints, timestamp)
                if player_data:
                    players_data.append(player_data)

                # 更新位置歷史
                self._update_position_history(person_id, keypoints)

            # 計算玩家間距離
            if len(players_data) >= 2:
                players_distance = self._calculate_players_distance(keypoints)

        return FrameData(
            frame_id=self.frame_count,
            timestamp=timestamp,
            players=players_data,
            players_distance=players_distance
        )

    def _detect_player_action(self, person_id: int, keypoints, timestamp: float) -> Optional[PlayerActionData]:
        """檢測單個玩家的動作"""
        try:
            person_keypoints = keypoints[person_id]

            # 檢查動作冷卻
            if person_id in self.player_cooldowns:
                if self.frame_count - self.player_cooldowns[person_id] < self.cooldown_frames:
                    return None

            # 計算手臂角度
            right_arm_angle = calculate_normalized_angle(
                keypoints, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], person_id)
            left_arm_angle = calculate_normalized_angle(
                keypoints, [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], person_id)

            # 計算肩寬
            shoulder_width = calculate_shoulder_width(keypoints, person_id)

            # 獲取身體中心
            body_center = self._get_body_center(person_keypoints)

            # 初始化動作數據
            action_data = PlayerActionData(
                player_id=person_id,
                action_type=ActionType.IDLE.value,
                attack_hand=None,
                punch_type=None,
                velocity=0.0,
                confidence=0.0,
                arm_angles={"right": right_arm_angle, "left": left_arm_angle},
                shoulder_width=shoulder_width,
                body_center=body_center,
                is_attacking=False,
                is_guarding=False,
                timestamp=timestamp
            )

            # 檢測防禦姿勢
            if self._detect_guard_pose(person_id, keypoints, right_arm_angle, left_arm_angle):
                action_data.action_type = ActionType.GUARD.value
                action_data.is_guarding = True
                self.player_cooldowns[person_id] = self.frame_count

            else:
                # 檢測攻擊動作
                punch_result = self._detect_punch_action(person_id, keypoints, right_arm_angle, left_arm_angle)
                if punch_result:
                    action_data.action_type = punch_result["action_type"]
                    action_data.attack_hand = punch_result["hand"]
                    action_data.punch_type = punch_result["punch_type"]
                    action_data.velocity = punch_result["velocity"]
                    action_data.confidence = punch_result["confidence"]
                    action_data.is_attacking = True
                    self.player_cooldowns[person_id] = self.frame_count

            return action_data

        except Exception as e:
            print(f"Error detecting player {person_id} action: {e}")
            return None

    def _detect_guard_pose(self, person_id: int, keypoints, right_arm_angle, left_arm_angle) -> bool:
        """檢測防禦姿勢"""
        if right_arm_angle is None or left_arm_angle is None:
            return False

        # 檢查雙手角度是否在防禦範圍內
        right_guard = self.guard_angle_min <= right_arm_angle <= self.guard_angle_max
        left_guard = self.guard_angle_min <= left_arm_angle <= self.guard_angle_max

        if not (right_guard and left_guard):
            return False

        # 額外檢查：手腕位置
        try:
            person_keypoints = keypoints[person_id]
            right_wrist = person_keypoints[RIGHT_WRIST]
            left_wrist = person_keypoints[LEFT_WRIST]
            neck = person_keypoints[NECK]

            if (right_wrist[2] > 0.3 and left_wrist[2] > 0.3 and neck[2] > 0.3):
                right_defensive = (right_wrist[0] > neck[0] - 100 and
                                   right_wrist[0] < neck[0] + 150 and
                                   right_wrist[1] > neck[1] - 50)
                left_defensive = (left_wrist[0] < neck[0] + 100 and
                                  left_wrist[0] > neck[0] - 150 and
                                  left_wrist[1] > neck[1] - 50)
                return right_defensive and left_defensive
        except (IndexError, TypeError):
            pass

        return False

    def _detect_punch_action(self, person_id: int, keypoints, right_arm_angle, left_arm_angle) -> Optional[Dict]:
        """檢測拳擊動作"""
        # 檢查右手攻擊
        right_punch = self._analyze_punch_motion(person_id, keypoints, "right", right_arm_angle)
        if right_punch:
            return right_punch

        # 檢查左手攻擊
        left_punch = self._analyze_punch_motion(person_id, keypoints, "left", left_arm_angle)
        if left_punch:
            return left_punch

        return None

    def _analyze_punch_motion(self, person_id: int, keypoints, hand: str, arm_angle: float) -> Optional[Dict]:
        """分析拳擊動作類型，回傳包含置信度、速度與手臂資訊的字典"""
        if arm_angle is None:
            return None

        try:
            person_keypoints = keypoints[person_id]

            # 選擇手臂關鍵點索引
            if hand == "right":
                wrist_idx, elbow_idx, shoulder_idx = RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER
            else:
                wrist_idx, elbow_idx, shoulder_idx = LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER

            wrist = person_keypoints[wrist_idx]
            elbow = person_keypoints[elbow_idx]
            shoulder = person_keypoints[shoulder_idx]

            # 關鍵點置信度檢查
            if wrist[2] < 0.3 or elbow[2] < 0.3 or shoulder[2] < 0.3:
                return None

            # 計算手腕相對肩膀位置與手臂伸展距離
            wrist_x_offset = wrist[0] - shoulder[0]
            wrist_y_offset = wrist[1] - shoulder[1]
            arm_extension = math.sqrt(wrist_x_offset ** 2 + wrist_y_offset ** 2)

            # 計算手臂速度
            velocity = self._calculate_hand_velocity(person_id, hand, wrist[:2])

            # 關鍵點平均置信度
            keypoint_conf = (wrist[2] + elbow[2] + shoulder[2]) / 3.0

            # 標準化手臂伸展度 (依肩寬)
            shoulder_width = calculate_shoulder_width(keypoints, person_id)
            if shoulder_width and shoulder_width > 0:
                normalized_extension = arm_extension / shoulder_width
                # 改進：使用平滑的sigmoid函數替代線性限制
                ext_score = 1.0 / (1.0 + math.exp(-(normalized_extension - 1.5) * 3))
            else:
                ext_score = min(arm_extension / 120.0, 1.0)

            # 改進：速度分數使用更合理的閾值和平滑函數
            velocity_threshold = 0.5  # 提高速度閾值
            vel_score = 1.0 / (1.0 + math.exp(-(velocity - velocity_threshold) * 8))

            # 改進的角度評分系統
            angle_scores = self._calculate_angle_scores(arm_angle)

            # 檢查基本攻擊條件
            min_extension = 80  # 提高最小伸展要求
            if arm_extension < min_extension or velocity < 0.1:
                return None

            # 選擇最佳拳法
            best_punch = None
            best_confidence = 0

            for punch_name, config in self.punch_configs.items():
                # 檢查該拳法的特定條件
                punch_valid, position_bonus = self._validate_punch_type(
                    punch_name, arm_angle, wrist_x_offset, wrist_y_offset, hand
                )

                if not punch_valid:
                    continue

                # 計算該拳法的綜合置信度
                angle_score = angle_scores.get(punch_name, 0)

                # 改進：更平衡的權重分配
                base_confidence = (
                        keypoint_conf * 0.25 +  # 關鍵點置信度
                        ext_score * 0.25 +  # 伸展度
                        vel_score * 0.25 +  # 速度
                        angle_score * 0.15 +  # 角度匹配度
                        position_bonus * 0.10  # 位置加分
                )

                # 最終置信度不超過1.0
                final_confidence = min(base_confidence, 1.0)

                if final_confidence > best_confidence and final_confidence > 0.5:  # 提高最低置信度閾值
                    best_punch = {
                        "action_type": getattr(ActionType, f"PUNCH_{punch_name.upper()}").value,
                        "hand": hand,
                        "punch_type": punch_name,
                        "velocity": round(velocity, 3),
                        "confidence": round(final_confidence, 3),
                        "subscores": {
                            "keypoint_conf": round(keypoint_conf, 3),
                            "extension": round(ext_score, 3),
                            "velocity": round(vel_score, 3),
                            "angle_score": round(angle_score, 3),
                            "position_bonus": round(position_bonus, 3)
                        }
                    }
                    best_confidence = final_confidence

            return best_punch

        except (IndexError, TypeError):
            return None

    def _calculate_angle_scores(self, arm_angle: float) -> Dict[str, float]:
        """計算各種拳法的角度評分"""
        scores = {}

        # 直拳：角度應該較大（手臂伸直）
        straight_optimal = 150
        straight_tolerance = 30
        scores["straight"] = max(0, 1 - abs(arm_angle - straight_optimal) / straight_tolerance)

        # 勾拳：角度中等
        hook_optimal = 90
        hook_tolerance = 25
        scores["hook"] = max(0, 1 - abs(arm_angle - hook_optimal) / hook_tolerance)

        # 上勾拳：角度較小
        uppercut_optimal = 60
        uppercut_tolerance = 20
        scores["uppercut"] = max(0, 1 - abs(arm_angle - uppercut_optimal) / uppercut_tolerance)

        return scores

    def _validate_punch_type(self, punch_name: str, arm_angle: float,
                             wrist_x_offset: float, wrist_y_offset: float, hand: str) -> Tuple[bool, float]:
        """驗證特定拳法類型並計算位置加分"""
        position_bonus = 0.0

        if punch_name == "straight":
            # 直拳：手臂角度大，手腕向前伸展
            if arm_angle < 120:
                return False, 0.0

            # 位置加分：手腕在肩膀前方且相對平直
            if abs(wrist_y_offset) < 30:  # 高度差小
                position_bonus += 0.3
            if (hand == "right" and wrist_x_offset > 50) or (hand == "left" and wrist_x_offset < -50):
                position_bonus += 0.2

        elif punch_name == "hook":
            # 勾拳：中等角度，橫向動作
            if arm_angle < 60 or arm_angle > 130:
                return False, 0.0

            # 位置加分：手腕在側方且高度適中
            if abs(wrist_x_offset) > 40:  # 橫向位移
                position_bonus += 0.3
            if abs(wrist_y_offset) < 40:  # 高度控制
                position_bonus += 0.2

        elif punch_name == "uppercut":
            # 上勾拳：角度小，向上動作
            if arm_angle > 90:
                return False, 0.0

            # 位置加分：手腕向上且在身體前方
            if wrist_y_offset < -20:  # 向上
                position_bonus += 0.4
            if abs(wrist_x_offset) < 60:  # 不要太偏側
                position_bonus += 0.1

        return True, min(position_bonus, 0.5)  # 位置加分上限0.5


    def _update_position_history(self, person_id: int, keypoints):
        """更新位置歷史記錄"""
        if person_id not in self.position_history:
            self.position_history[person_id] = {"right": [], "left": []}

        try:
            person_keypoints = keypoints[person_id]

            for hand, wrist_idx in [("right", RIGHT_WRIST), ("left", LEFT_WRIST)]:
                wrist = person_keypoints[wrist_idx]
                if wrist[2] > 0.3:
                    history = self.position_history[person_id][hand]
                    history.append((wrist[0], wrist[1], time.time()))

                    if len(history) > 8:
                        history.pop(0)

        except (IndexError, TypeError):
            pass

    def _calculate_hand_velocity(self, person_id: int, hand: str, current_pos: Tuple[float, float]) -> float:
        if person_id not in self.position_history or hand not in self.position_history[person_id]:
            return 0.0

        history = self.position_history[person_id][hand]
        if len(history) < 3:
            return 0.0

        recent_positions = history[-3:]
        total_distance = 0.0
        total_time = 0.0

        for i in range(1, len(recent_positions)):
            prev = recent_positions[i - 1]
            curr = recent_positions[i]
            distance = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
            time_diff = max(curr[2] - prev[2], 1e-3)  # 防止除以0
            total_distance += distance
            total_time += time_diff

        velocity = total_distance / total_time
        # 平滑化，限制過高速度
        velocity = min(velocity, 3.0)  # px/s 最大值，可根據測試調整
        return velocity

    def _calculate_players_distance(self, keypoints) -> Optional[float]:
        """計算兩個玩家之間的距離"""
        try:
            if len(keypoints) < 2:
                return None

            center_0 = self._get_body_center(keypoints[0])
            center_1 = self._get_body_center(keypoints[1])

            if center_0 is None or center_1 is None:
                return None

            distance = math.sqrt((center_0[0] - center_1[0]) ** 2 + (center_0[1] - center_1[1]) ** 2)
            return distance

        except (IndexError, TypeError):
            return None

    def _get_body_center(self, person_keypoints) -> Optional[Tuple[float, float]]:
        """獲取身體中心點"""
        try:
            neck = person_keypoints[NECK]
            hip = person_keypoints[MID_HIP]

            if neck[2] > 0.3 and hip[2] > 0.3:
                return ((neck[0] + hip[0]) / 2, (neck[1] + hip[1]) / 2)
            elif neck[2] > 0.3:
                return (neck[0], neck[1])
            elif hip[2] > 0.3:
                return (hip[0], hip[1])

        except (IndexError, TypeError):
            pass

        return None

    def reset_history(self):
        """重置歷史記錄（用於新場景開始）"""
        self.position_history.clear()
        self.player_cooldowns.clear()
        self.frame_count = 0
        print("Boxing detector history reset")

    def get_detection_stats(self) -> Dict[str, Any]:
        """獲取檢測統計信息"""
        return {
            "total_frames": self.frame_count,
            "tracked_players": len(self.position_history),
            "punch_configs": {name: config.name for name, config in self.punch_configs.items()},
            "guard_angle_range": (self.guard_angle_min, self.guard_angle_max),
            "cooldown_frames": self.cooldown_frames
        }

    def detect_comprehensive_actions(self, keypoints):
        """同時檢測攻擊和防禦動作"""
        # 檢測攻擊動作
        attack_frame_data = self.detect_actions(keypoints)

        # 檢測防禦動作
        defense_frame_data = self.defense_detector.detect_defense_actions(keypoints)

        return attack_frame_data, defense_frame_data


class BoxingVisualizer:
    """拳擊動作可視化器 - 獨立的視覺化模組"""

    def __init__(self, show_skeleton=True, show_debug=True):
        self.show_skeleton = show_skeleton
        self.show_debug = show_debug
        self.last_frame_time = time.time()

    def draw_debug_frame(self, frame, keypoints, frame_data: FrameData):
        """繪製調試信息幀"""
        if not self.show_debug:
            return frame

        result_frame = frame.copy()
        height, width = frame.shape[:2]

        # 繪製骨架
        if self.show_skeleton and keypoints is not None:
            for person_id in range(min(len(keypoints), 2)):
                self._draw_skeleton(result_frame, keypoints[person_id], person_id)

        # 繪製動作信息
        for player_data in frame_data.players:
            self._draw_action_info(result_frame, keypoints, player_data)

        # 繪製系統狀態
        self._draw_system_status(result_frame, frame_data, width, height)

        return result_frame

    def _draw_skeleton(self, frame, person_keypoints, person_id):
        """繪製人體骨架"""
        color = (0, 255, 0) if person_id == 0 else (255, 0, 0)

        # 骨架連接定義
        skeleton_connections = [
            (NECK, RIGHT_SHOULDER), (NECK, LEFT_SHOULDER),
            (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
            (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
            (NECK, MID_HIP), (MID_HIP, RIGHT_HIP), (MID_HIP, LEFT_HIP),
            (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
            (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE)
        ]

        # 繪製骨架線條
        for start_idx, end_idx in skeleton_connections:
            start_point = person_keypoints[start_idx]
            end_point = person_keypoints[end_idx]

            if start_point[2] > 0.3 and end_point[2] > 0.3:
                cv2.line(frame,
                         (int(start_point[0]), int(start_point[1])),
                         (int(end_point[0]), int(end_point[1])),
                         color, 2)

        # 繪製關鍵點
        for keypoint in person_keypoints:
            if keypoint[2] > 0.3:
                cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 4, color, -1)

    def _draw_action_info(self, frame, keypoints, player_data: PlayerActionData):
        """繪製動作信息"""
        if keypoints is None or player_data.player_id >= len(keypoints):
            return

        person_keypoints = keypoints[player_data.player_id]
        color = (0, 255, 0) if player_data.player_id == 0 else (255, 0, 0)

        # 獲取頭部位置
        head_pos = None
        if person_keypoints[NOSE][2] > 0.3:
            head_pos = (int(person_keypoints[NOSE][0]), int(person_keypoints[NOSE][1]) - 50)
        elif person_keypoints[NECK][2] > 0.3:
            head_pos = (int(person_keypoints[NECK][0]), int(person_keypoints[NECK][1]) - 30)

        if head_pos:
            # 顯示動作類型
            action_text = player_data.action_type.replace('_', ' ').title()
            cv2.putText(frame, action_text, head_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 顯示攻擊手和拳法類型
            if player_data.is_attacking and player_data.attack_hand:
                hand_text = f"{player_data.attack_hand.upper()} {player_data.punch_type.upper()}"
                cv2.putText(frame, hand_text,
                            (head_pos[0], head_pos[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 顯示置信度
                conf_text = f"Conf: {player_data.confidence:.2f}"
                cv2.putText(frame, conf_text,
                            (head_pos[0], head_pos[1] + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_system_status(self, frame, frame_data: FrameData, width, height):
        """繪製系統狀態信息"""
        # 背景框
        cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 100), (255, 255, 255), 2)

        # 幀信息
        cv2.putText(frame, f"Frame: {frame_data.frame_id}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 玩家數量
        cv2.putText(frame, f"Players: {len(frame_data.players)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 玩家間距離
        if frame_data.players_distance is not None:
            cv2.putText(frame, f"Distance: {frame_data.players_distance:.1f}px", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS計算
        current_time = time.time()
        fps = 1.0 / max(current_time - self.last_frame_time, 0.001)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.last_frame_time = current_time


class ComprehensiveVisualizer:
    """綜合攻擊和防禦動作可視化器"""

    def __init__(self, show_skeleton=True, show_debug=True):
        self.attack_visualizer = BoxingVisualizer(show_skeleton, show_debug)
        self.defense_visualizer = BoxingDefenseVisualizer(show_skeleton, show_debug)
        self.last_frame_time = time.time()

    def draw_comprehensive_frame(self, frame, keypoints, attack_data, defense_data):
        """繪製綜合動作信息"""
        result_frame = frame.copy()
        height, width = frame.shape[:2]

        # 繪製骨架
        if keypoints is not None:
            for person_id in range(min(len(keypoints), 2)):
                self._draw_skeleton(result_frame, keypoints[person_id], person_id)

        # 繪製每個玩家的狀態信息
        self._draw_player_actions(result_frame, keypoints, attack_data, defense_data)

        # 繪製系統狀態
        self._draw_system_status(result_frame, attack_data, defense_data, width, height)

        return result_frame

    def _draw_skeleton(self, frame, person_keypoints, person_id):
        """繪製人體骨架"""
        color = (0, 255, 0) if person_id == 0 else (255, 0, 0)

        skeleton_connections = [
            (NECK, RIGHT_SHOULDER), (NECK, LEFT_SHOULDER),
            (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
            (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
            (NECK, MID_HIP), (MID_HIP, RIGHT_HIP), (MID_HIP, LEFT_HIP),
            (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
            (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE)
        ]

        for start_idx, end_idx in skeleton_connections:
            start_point = person_keypoints[start_idx]
            end_point = person_keypoints[end_idx]

            if start_point[2] > 0.3 and end_point[2] > 0.3:
                cv2.line(frame,
                         (int(start_point[0]), int(start_point[1])),
                         (int(end_point[0]), int(end_point[1])),
                         color, 2)

        for keypoint in person_keypoints:
            if keypoint[2] > 0.3:
                cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 4, color, -1)

    def _draw_player_actions(self, frame, keypoints, attack_data, defense_data):
        """繪製玩家動作信息"""
        if keypoints is None:
            return

        # 為每個玩家繪製信息
        for person_id in range(min(len(keypoints), 2)):
            person_keypoints = keypoints[person_id]
            head_pos = self._get_head_position(person_keypoints)

            if not head_pos:
                continue

            color = (0, 255, 0) if person_id == 0 else (255, 0, 0)
            y_offset = 0

            # 查找該玩家的攻擊數據
            attack_player = None
            for player in attack_data.players:
                if player.player_id == person_id:
                    attack_player = player
                    break

            # 查找該玩家的防禦數據
            defense_player = None
            for player in defense_data.players:
                if player.player_id == person_id:
                    defense_player = player
                    break

            # 顯示攻擊動作
            if attack_player and attack_player.is_attacking:
                attack_text = f"ATTACK: {attack_player.punch_type.upper()} ({attack_player.attack_hand.upper()})"
                cv2.putText(frame, attack_text, (head_pos[0], head_pos[1] + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 25

                conf_text = f"Conf: {attack_player.confidence:.2f}"
                cv2.putText(frame, conf_text, (head_pos[0], head_pos[1] + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_offset += 25

            # 顯示防禦動作
            if defense_player and defense_player.is_defending:
                defense_text = f"DEFENSE: {defense_player.defense_type.replace('_', ' ').upper()}"
                cv2.putText(frame, defense_text, (head_pos[0], head_pos[1] + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25

                if defense_player.guard_hand:
                    hand_text = f"Guard: {defense_player.guard_hand.upper()}"
                    cv2.putText(frame, hand_text, (head_pos[0], head_pos[1] + y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    y_offset += 25

            # 顯示閃避動作
            if defense_player and defense_player.is_dodging:
                dodge_text = f"DODGE: {defense_player.dodge_direction.upper()}"
                cv2.putText(frame, dodge_text, (head_pos[0], head_pos[1] + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 25

            # 如果沒有特殊動作，顯示 IDLE
            if ((not attack_player or not attack_player.is_attacking) and
                    (not defense_player or (not defense_player.is_defending and not defense_player.is_dodging))):
                cv2.putText(frame, "IDLE", (head_pos[0], head_pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _get_head_position(self, person_keypoints):
        """獲取頭部位置"""
        if person_keypoints[NOSE][2] > 0.3:
            return (int(person_keypoints[NOSE][0]), int(person_keypoints[NOSE][1]) - 50)
        elif person_keypoints[NECK][2] > 0.3:
            return (int(person_keypoints[NECK][0]), int(person_keypoints[NECK][1]) - 30)
        return None

    def _draw_system_status(self, frame, attack_data, defense_data, width, height):
        """繪製系統狀態信息"""
        # 背景框
        cv2.rectangle(frame, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 140), (255, 255, 255), 2)

        # 幀信息
        cv2.putText(frame, f"Frame: {attack_data.frame_id}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 玩家數量
        cv2.putText(frame, f"Players: {len(attack_data.players)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 攻擊統計
        attacking_count = sum(1 for p in attack_data.players if p.is_attacking)
        cv2.putText(frame, f"Attacking: {attacking_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 防禦統計
        defending_count = sum(1 for p in defense_data.players if p.is_defending)
        cv2.putText(frame, f"Defending: {defending_count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 玩家間距離
        if attack_data.players_distance is not None:
            cv2.putText(frame, f"Distance: {attack_data.players_distance:.1f}px", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS計算
        current_time = time.time()
        fps = 1.0 / max(current_time - self.last_frame_time, 0.001)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.last_frame_time = current_time


def run_standalone_detection(camera_index=0, ground_truth_action="test"):
    """獨立運行綜合動作檢測（攻擊+防禦）"""
    print("=== Running Comprehensive Boxing Detection (Attack + Defense) ===")
    print(f"=== Running Detection for: {ground_truth_action} ===")

    detector = BoxingActionDetector()
    visualizer = ComprehensiveVisualizer()

    # --- 開始：日誌設定 ---
    # 產生帶有時間戳和動作名稱的日誌檔名
    log_filename = f"log_{ground_truth_action}_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    # 打開檔案準備寫入
    log_file = open(log_filename, 'w', newline='', encoding='utf-8')

    # 建立 CSV 寫入器
    log_writer = csv.writer(log_file)

    # 寫入試算表之實驗數據標頭，可擴充想紀錄的內容
    header = [
        "timestamp", "frame_id", "person_id", "ground_truth_action",
        "detected_action", "confidence", "is_correct"
    ]
    log_writer.writerow(header)
    # --- 結束：日誌設定 ---

    try:
        for frame_id, keypoints_data, frame in get_keypoints_stream(video_source=camera_index):

            # 同時檢測攻擊和防禦動作
            attack_data, defense_data = detector.detect_comprehensive_actions(keypoints_data)

            # --- 開始：寫入日誌 ---
            # 遍歷每個被偵測到的人
            num_players = max(len(attack_data.players), len(defense_data.players))
            for person_id in range(num_players):

                # 找到對應玩家的攻擊和防禦數據
                attack_player = next((p for p in attack_data.players if p.player_id == person_id), None)
                defense_player = next((p for p in defense_data.players if p.player_id == person_id), None)

                detected_action = "idle"
                confidence = 0.0

                # 優先判斷高置信度的動作
                if attack_player and attack_player.is_attacking:
                    detected_action = attack_player.punch_type
                    confidence = attack_player.confidence
                elif defense_player and defense_player.is_defending:
                    detected_action = defense_player.defense_type
                    confidence = defense_player.confidence
                elif defense_player and defense_player.is_dodging:
                    detected_action = defense_player.dodge_direction
                    confidence = defense_player.confidence

                # 判斷是否正確
                # 注意：這裡的邏輯需要根據你的測試方式調整
                # 簡單假設：如果辨識出的動作名稱包含在 "ground_truth" 中，就視為正確
                # 例如，ground_truth="straight_punch", detected_action="straight" -> 正確
                # 預設為不正確
                is_correct = 0
                # 精確匹配
                if ground_truth_action.lower() == detected_action.lower():
                    is_correct = 1
                # 處理特殊情況，例如測試 "punch" 時，任何一種拳都算對
                elif ground_truth_action.lower() == "punch" and detected_action in ["straight", "hook", "uppercut"]:
                    is_correct = 1
                # 處理防禦的總稱
                elif ground_truth_action.lower() == "guard" and "guard" in detected_action.lower():
                    is_correct = 1
                # 處理閃避的總稱
                elif ground_truth_action.lower() == "dodge" and "dodge" in detected_action.lower():
                    is_correct = 1

                # 準備寫入的單行數據
                log_row = [
                    time.time(),
                    frame_id,
                    person_id,
                    ground_truth_action,
                    detected_action,
                    confidence,
                    is_correct
                ]

                # 寫入 CSV
                log_writer.writerow(log_row)
            # --- 結束：寫入日誌 ---

            # 可視化結果
            result_frame = visualizer.draw_comprehensive_frame(frame, keypoints_data, attack_data, defense_data)

            # 在終端打印檢測結果
            for player_id in range(min(len(attack_data.players), len(defense_data.players))):
                # 查找攻擊數據
                attack_player = None
                for p in attack_data.players:
                    if p.player_id == player_id:
                        attack_player = p
                        break

                # 查找防禦數據
                defense_player = None
                for p in defense_data.players:
                    if p.player_id == player_id:
                        defense_player = p
                        break

                # 打印動作狀態
                actions = []

                if attack_player and attack_player.is_attacking:
                    actions.append(
                        f"ATTACK: {attack_player.punch_type} with {attack_player.attack_hand} (conf: {attack_player.confidence:.2f})")

                if defense_player and defense_player.is_defending:
                    actions.append(f"DEFENSE: {defense_player.defense_type} (conf: {defense_player.confidence:.2f})")

                if defense_player and defense_player.is_dodging:
                    actions.append(f"DODGE: {defense_player.dodge_direction}")

                if actions:
                    print(f"Player {player_id}: {' | '.join(actions)}")

            # 顯示結果
            cv2.imshow('Boxing Detection - Attack & Defense', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Detection stopped by user")
    finally:
        # --- 開始：關閉日誌檔案 ---
        if log_file:
            log_file.close()
            print(f"Log saved to: {log_filename}")
        # --- 結束：關閉日誌檔案 ---
        cv2.destroyAllWindows()
        print("Comprehensive detection completed")


if __name__ == "__main__":
    """主函數---僅用於測試動作檢測"""
    import argparse

    parser = argparse.ArgumentParser(description='Boxing Action Detection - Standalone Mode')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')

    parser.add_argument(
        '--action',
        type=str,
        default="idle_test",
        help='Specify the ground truth action for this test run (e.g., "straight_punch", "high_guard")'
    )

    args = parser.parse_args()

    # 運行獨立檢測模式
    run_standalone_detection(camera_index=args.camera, ground_truth_action=args.action)
