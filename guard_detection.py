# 修改後的boxing_guard.py - 專注於拳擊防禦動作辨識
import math
import time
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from collections import deque
import cv2

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
    print("Warning: angle.py not found, using built-in angle functions")

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


# 定義拳擊防禦動作類型
class BoxingDefenseType(Enum):
    IDLE = "idle"
    HIGH_GUARD = "high_guard"  # 高位防禦 - 保護頭部
    MID_GUARD = "mid_guard"  # 中位防禦 - 保護身體
    LOW_GUARD = "low_guard"  # 低位防禦 - 保護下盤
    CROSS_GUARD = "cross_guard"  # 交叉防禦
    PARRY_LEFT = "parry_left"  # 左手撥擋
    PARRY_RIGHT = "parry_right"  # 右手撥擋
    SHELL_GUARD = "shell_guard"  # 貝殼式防守
    PEEK_A_BOO = "peek_a_boo"  # 窺視拳防守
    DODGE_LEFT = "dodge_left"  # 左閃
    DODGE_RIGHT = "dodge_right"  # 右閃
    DUCK = "duck"  # 蹲閃
    SLIP_LEFT = "slip_left"  # 左滑步
    SLIP_RIGHT = "slip_right"  # 右滑步


# 防禦配置的資料型別
@dataclass
class BoxingGuardConfig:
    name: str
    angle_min: float
    angle_max: float
    effectiveness: float  # 防禦效果 0-1
    stamina_cost: float  # 體力消耗 0-1
    coverage_areas: List[str]  # 保護區域
    counter_opportunity: float  # 反擊機會 0-1


# 玩家防禦動作數據
@dataclass
class PlayerDefenseData:
    player_id: int
    defense_type: str
    guard_hand: Optional[str]  # "left", "right", "both"
    velocity: float
    confidence: float
    arm_angles: Dict[str, Optional[float]]
    shoulder_width: Optional[float]
    body_center: Optional[Tuple[float, float]]
    is_defending: bool
    is_dodging: bool
    dodge_direction: Optional[str]
    defense_effectiveness: float
    coverage_areas: List[str]
    stamina_cost: float
    counter_ready: bool
    stance_stability: float  # 站位穩定度
    guard_position: Dict[str, float]  # 防禦位置評分
    timestamp: float


# 單幀數據
@dataclass
class DefenseFrameData:
    frame_id: int
    timestamp: float
    players: List[PlayerDefenseData]
    players_distance: Optional[float]


class BoxingDefenseDetector:
    """拳擊防禦動作檢測器"""

    def __init__(self):
        self.frame_count = 0

        # 拳擊防禦配置
        self.defense_configs = {
            "high_guard": BoxingGuardConfig(
                "高位防禦", 45, 85, 0.85, 0.1,
                ["頭部", "太陽穴", "下巴"], 0.3
            ),
            "mid_guard": BoxingGuardConfig(
                "中位防禦", 70, 120, 0.80, 0.08,
                ["胸部", "肋骨", "心臟"], 0.4
            ),
            "low_guard": BoxingGuardConfig(
                "低位防禦", 100, 160, 0.70, 0.06,
                ["腹部", "肝臟", "腎臟"], 0.2
            ),
            "cross_guard": BoxingGuardConfig(
                "交叉防禦", 60, 100, 0.90, 0.15,
                ["頭部", "胸部"], 0.5
            ),
            "parry": BoxingGuardConfig(
                "撥擋", 120, 180, 0.95, 0.2,
                ["偏轉攻擊"], 0.8
            ),
            "shell_guard": BoxingGuardConfig(
                "貝殼防守", 30, 70, 0.88, 0.12,
                ["頭部", "肩膀"], 0.6
            ),
            "peek_a_boo": BoxingGuardConfig(
                "窺視拳防守", 50, 90, 0.82, 0.14,
                ["頭部", "上身"], 0.4
            )
        }

        # 移動防禦參數
        self.dodge_threshold = 0.4
        self.slip_threshold = 0.3
        self.duck_threshold = 0.25

        # 位置歷史記錄
        self.position_history = {}
        self.defense_history = {}
        self.history_length = 6

        # 動作冷卻
        self.player_cooldowns = {}
        self.cooldown_frames = 8

        print("=== Boxing Defense Detector Initialized ===")

    def detect_defense_actions(self, keypoints) -> DefenseFrameData:
        """檢測拳擊防禦動作"""
        self.frame_count += 1
        timestamp = time.time()

        players_data = []
        players_distance = None

        if keypoints is not None and len(keypoints) > 0:
            num_persons = min(len(keypoints), 2)

            for person_id in range(num_persons):
                player_data = self._detect_player_defense(person_id, keypoints, timestamp)
                if player_data:
                    players_data.append(player_data)

                self._update_position_history(person_id, keypoints)

            if len(players_data) >= 2:
                players_distance = self._calculate_players_distance(keypoints)

        return DefenseFrameData(
            frame_id=self.frame_count,
            timestamp=timestamp,
            players=players_data,
            players_distance=players_distance
        )

    def _detect_player_defense(self, person_id: int, keypoints, timestamp: float) -> Optional[PlayerDefenseData]:
        """檢測單個玩家的防禦動作"""
        try:
            person_keypoints = keypoints[person_id]

            # 檢查冷卻時間
            if person_id in self.player_cooldowns:
                if self.frame_count - self.player_cooldowns[person_id] < self.cooldown_frames:
                    return self._create_idle_defense_data(person_id, keypoints, timestamp)

            # 計算手臂角度
            right_arm_angle = calculate_normalized_angle(
                keypoints, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], person_id)
            left_arm_angle = calculate_normalized_angle(
                keypoints, [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], person_id)

            # 基本數據
            shoulder_width = calculate_shoulder_width(keypoints, person_id)
            body_center = self._get_body_center(person_keypoints)
            stance_stability = self._calculate_stance_stability(person_id, keypoints)

            # 初始化防禦數據
            defense_data = PlayerDefenseData(
                player_id=person_id,
                defense_type=BoxingDefenseType.IDLE.value,
                guard_hand=None,
                velocity=0.0,
                confidence=0.0,
                arm_angles={"right": right_arm_angle, "left": left_arm_angle},
                shoulder_width=shoulder_width,
                body_center=body_center,
                is_defending=False,
                is_dodging=False,
                dodge_direction=None,
                defense_effectiveness=0.0,
                coverage_areas=[],
                stamina_cost=0.0,
                counter_ready=False,
                stance_stability=stance_stability,
                guard_position={},
                timestamp=timestamp
            )

            # 優先檢測移動防禦（閃避、滑步等）
            movement_defense = self._detect_movement_defense(person_id, keypoints)
            if movement_defense:
                self._apply_defense_result(defense_data, movement_defense)
                self.player_cooldowns[person_id] = self.frame_count
                return defense_data

            # 檢測靜態防禦姿勢
            static_defense = self._detect_static_defense(person_id, keypoints, right_arm_angle, left_arm_angle)
            if static_defense:
                self._apply_defense_result(defense_data, static_defense)
                self.player_cooldowns[person_id] = self.frame_count
                return defense_data

            return defense_data

        except Exception as e:
            print(f"Error detecting player {person_id} defense: {e}")
            return None

    def _detect_static_defense(self, person_id: int, keypoints, right_arm_angle, left_arm_angle) -> Optional[Dict]:
        """檢測靜態防禦姿勢"""
        if right_arm_angle is None or left_arm_angle is None:
            return None

        try:
            person_keypoints = keypoints[person_id]

            # 獲取關鍵點
            right_wrist = person_keypoints[RIGHT_WRIST]
            left_wrist = person_keypoints[LEFT_WRIST]
            right_elbow = person_keypoints[RIGHT_ELBOW]
            left_elbow = person_keypoints[LEFT_ELBOW]
            right_shoulder = person_keypoints[RIGHT_SHOULDER]
            left_shoulder = person_keypoints[LEFT_SHOULDER]
            neck = person_keypoints[NECK]

            # 置信度檢查
            key_points = [right_wrist, left_wrist, right_elbow, left_elbow,
                          right_shoulder, left_shoulder, neck]
            if any(point[2] < 0.3 for point in key_points):
                return None

            # 計算手部位置相對於身體的高度
            right_hand_height = right_wrist[1] - neck[1]
            left_hand_height = left_wrist[1] - neck[1]
            avg_hand_height = (right_hand_height + left_hand_height) / 2

            # 計算手臂速度
            velocity = self._calculate_arm_velocity(person_id, "both",
                                                    [(right_wrist[0], right_wrist[1]),
                                                     (left_wrist[0], left_wrist[1])])

            # 關鍵點平均置信度
            keypoint_confidence = sum(point[2] for point in key_points) / len(key_points)

            defense_results = []

            # 檢測撥擋動作 - 優先級最高
            parry_result = self._detect_parry_action(
                person_keypoints, right_arm_angle, left_arm_angle, velocity, keypoint_confidence
            )
            if parry_result:
                defense_results.append(parry_result)

            # 檢測交叉防禦
            cross_guard_result = self._detect_cross_guard(
                person_keypoints, right_arm_angle, left_arm_angle, keypoint_confidence
            )
            if cross_guard_result:
                defense_results.append(cross_guard_result)

            # 檢測貝殼式防守
            shell_guard_result = self._detect_shell_guard(
                person_keypoints, right_arm_angle, left_arm_angle, keypoint_confidence
            )
            if shell_guard_result:
                defense_results.append(shell_guard_result)

            # 根據手部高度檢測位置防禦
            position_defenses = self._detect_position_defenses(
                avg_hand_height, right_arm_angle, left_arm_angle, keypoint_confidence, velocity
            )
            defense_results.extend(position_defenses)

            # 返回最佳結果
            if defense_results:
                best_defense = max(defense_results, key=lambda x: x["confidence"])
                if best_defense["confidence"] > 0.5:
                    return best_defense

        except (IndexError, TypeError):
            pass

        return None

    def _detect_parry_action(self, person_keypoints, right_arm_angle, left_arm_angle,
                             velocity, keypoint_confidence) -> Optional[Dict]:
        """檢測撥擋動作"""
        parry_results = []

        # 右手撥擋
        if right_arm_angle and right_arm_angle > 120:
            right_wrist = person_keypoints[RIGHT_WRIST]
            right_shoulder = person_keypoints[RIGHT_SHOULDER]

            # 檢查手臂伸展和速度
            arm_extension = abs(right_wrist[0] - right_shoulder[0])
            shoulder_width = calculate_shoulder_width([person_keypoints], 0)

            if shoulder_width and arm_extension > shoulder_width * 0.6 and velocity > 0.8:
                config = self.defense_configs["parry"]
                confidence = self._calculate_defense_confidence(
                    keypoint_confidence, right_arm_angle, None, config, velocity, 0.3
                )
                parry_results.append({
                    "defense_type": BoxingDefenseType.PARRY_RIGHT.value,
                    "guard_hand": "right",
                    "confidence": confidence,
                    "effectiveness": config.effectiveness,
                    "coverage_areas": config.coverage_areas,
                    "stamina_cost": config.stamina_cost,
                    "counter_ready": True,
                    "velocity": velocity
                })

        # 左手撥擋
        if left_arm_angle and left_arm_angle > 120:
            left_wrist = person_keypoints[LEFT_WRIST]
            left_shoulder = person_keypoints[LEFT_SHOULDER]

            arm_extension = abs(left_wrist[0] - left_shoulder[0])
            shoulder_width = calculate_shoulder_width([person_keypoints], 0)

            if shoulder_width and arm_extension > shoulder_width * 0.6 and velocity > 0.8:
                config = self.defense_configs["parry"]
                confidence = self._calculate_defense_confidence(
                    keypoint_confidence, None, left_arm_angle, config, velocity, 0.3
                )
                parry_results.append({
                    "defense_type": BoxingDefenseType.PARRY_LEFT.value,
                    "guard_hand": "left",
                    "confidence": confidence,
                    "effectiveness": config.effectiveness,
                    "coverage_areas": config.coverage_areas,
                    "stamina_cost": config.stamina_cost,
                    "counter_ready": True,
                    "velocity": velocity
                })

        return max(parry_results, key=lambda x: x["confidence"]) if parry_results else None

    def _detect_cross_guard(self, person_keypoints, right_arm_angle, left_arm_angle,
                            keypoint_confidence) -> Optional[Dict]:
        """檢測交叉防禦"""
        if not (right_arm_angle and left_arm_angle):
            return None

        right_wrist = person_keypoints[RIGHT_WRIST]
        left_wrist = person_keypoints[LEFT_WRIST]
        right_shoulder = person_keypoints[RIGHT_SHOULDER]
        left_shoulder = person_keypoints[LEFT_SHOULDER]

        # 計算手腕交叉距離
        wrist_distance = abs(right_wrist[0] - left_wrist[0])
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

        # 檢查是否交叉且角度合適
        if (wrist_distance < shoulder_width * 0.7 and
                self._angle_in_range(right_arm_angle, self.defense_configs["cross_guard"]) and
                self._angle_in_range(left_arm_angle, self.defense_configs["cross_guard"])):
            config = self.defense_configs["cross_guard"]
            confidence = self._calculate_defense_confidence(
                keypoint_confidence, right_arm_angle, left_arm_angle, config, 0.0, 0.1
            )

            return {
                "defense_type": BoxingDefenseType.CROSS_GUARD.value,
                "guard_hand": "both",
                "confidence": confidence,
                "effectiveness": config.effectiveness,
                "coverage_areas": config.coverage_areas,
                "stamina_cost": config.stamina_cost,
                "counter_ready": False,
                "velocity": 0.0
            }

        return None

    def _detect_shell_guard(self, person_keypoints, right_arm_angle, left_arm_angle,
                            keypoint_confidence) -> Optional[Dict]:
        """檢測貝殼式防守"""
        if not (right_arm_angle and left_arm_angle):
            return None

        # 貝殼式防守特徵：雙臂貼近身體，肘部向下
        right_elbow = person_keypoints[RIGHT_ELBOW]
        left_elbow = person_keypoints[LEFT_ELBOW]
        right_shoulder = person_keypoints[RIGHT_SHOULDER]
        left_shoulder = person_keypoints[LEFT_SHOULDER]

        # 檢查肘部是否貼近身體
        right_elbow_close = abs(right_elbow[0] - right_shoulder[0]) < 50
        left_elbow_close = abs(left_elbow[0] - left_shoulder[0]) < 50

        if (right_elbow_close and left_elbow_close and
                self._angle_in_range(right_arm_angle, self.defense_configs["shell_guard"]) and
                self._angle_in_range(left_arm_angle, self.defense_configs["shell_guard"])):
            config = self.defense_configs["shell_guard"]
            confidence = self._calculate_defense_confidence(
                keypoint_confidence, right_arm_angle, left_arm_angle, config, 0.0, 0.05
            )

            return {
                "defense_type": BoxingDefenseType.SHELL_GUARD.value,
                "guard_hand": "both",
                "confidence": confidence,
                "effectiveness": config.effectiveness,
                "coverage_areas": config.coverage_areas,
                "stamina_cost": config.stamina_cost,
                "counter_ready": True,
                "velocity": 0.0
            }

        return None

    def _detect_position_defenses(self, avg_hand_height, right_arm_angle, left_arm_angle,
                                  keypoint_confidence, velocity) -> List[Dict]:
        """根據手部位置檢測防禦類型"""
        defenses = []

        # 高位防禦
        if avg_hand_height < -40:
            if (self._angle_in_range(right_arm_angle, self.defense_configs["high_guard"]) and
                    self._angle_in_range(left_arm_angle, self.defense_configs["high_guard"])):
                config = self.defense_configs["high_guard"]
                confidence = self._calculate_defense_confidence(
                    keypoint_confidence, right_arm_angle, left_arm_angle, config, velocity, 0.05
                )
                defenses.append({
                    "defense_type": BoxingDefenseType.HIGH_GUARD.value,
                    "guard_hand": "both",
                    "confidence": confidence,
                    "effectiveness": config.effectiveness,
                    "coverage_areas": config.coverage_areas,
                    "stamina_cost": config.stamina_cost,
                    "counter_ready": False,
                    "velocity": velocity
                })

        # 中位防禦
        elif -20 <= avg_hand_height <= 40:
            if (self._angle_in_range(right_arm_angle, self.defense_configs["mid_guard"]) and
                    self._angle_in_range(left_arm_angle, self.defense_configs["mid_guard"])):
                config = self.defense_configs["mid_guard"]
                confidence = self._calculate_defense_confidence(
                    keypoint_confidence, right_arm_angle, left_arm_angle, config, velocity, 0.03
                )
                defenses.append({
                    "defense_type": BoxingDefenseType.MID_GUARD.value,
                    "guard_hand": "both",
                    "confidence": confidence,
                    "effectiveness": config.effectiveness,
                    "coverage_areas": config.coverage_areas,
                    "stamina_cost": config.stamina_cost,
                    "counter_ready": False,
                    "velocity": velocity
                })

        # 低位防禦
        elif avg_hand_height > 40:
            if (self._angle_in_range(right_arm_angle, self.defense_configs["low_guard"]) and
                    self._angle_in_range(left_arm_angle, self.defense_configs["low_guard"])):
                config = self.defense_configs["low_guard"]
                confidence = self._calculate_defense_confidence(
                    keypoint_confidence, right_arm_angle, left_arm_angle, config, velocity, 0.02
                )
                defenses.append({
                    "defense_type": BoxingDefenseType.LOW_GUARD.value,
                    "guard_hand": "both",
                    "confidence": confidence,
                    "effectiveness": config.effectiveness,
                    "coverage_areas": config.coverage_areas,
                    "stamina_cost": config.stamina_cost,
                    "counter_ready": False,
                    "velocity": velocity
                })

        return defenses

    def _detect_movement_defense(self, person_id: int, keypoints) -> Optional[Dict]:
        """檢測移動防禦（閃避、滑步等）"""
        try:
            person_keypoints = keypoints[person_id]

            neck = person_keypoints[NECK]
            hip = person_keypoints[MID_HIP]
            right_shoulder = person_keypoints[RIGHT_SHOULDER]
            left_shoulder = person_keypoints[LEFT_SHOULDER]

            if any(point[2] < 0.3 for point in [neck, hip, right_shoulder, left_shoulder]):
                return None

            # 計算身體運動速度
            body_velocity = self._calculate_body_velocity(person_id, keypoints)

            if body_velocity < 0.2:  # 運動太慢不算閃避
                return None

            # 計算身體中心和肩膀中心
            body_center_x = (neck[0] + hip[0]) / 2
            shoulder_center_x = (right_shoulder[0] + left_shoulder[0]) / 2
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

            if shoulder_width < 10:
                return None

            # 檢測側向閃避
            lateral_offset = body_center_x - shoulder_center_x
            normalized_offset = abs(lateral_offset) / shoulder_width

            if normalized_offset > self.dodge_threshold and body_velocity > 0.4:
                direction = "left" if lateral_offset < 0 else "right"
                action_type = BoxingDefenseType.DODGE_LEFT if direction == "left" else BoxingDefenseType.DODGE_RIGHT

                confidence = min(0.7 + normalized_offset * 0.2 + body_velocity * 0.1, 1.0)

                return {
                    "defense_type": action_type.value,
                    "guard_hand": None,
                    "confidence": confidence,
                    "effectiveness": 0.9,
                    "coverage_areas": ["全身閃避"],
                    "stamina_cost": 0.25,
                    "counter_ready": True,
                    "velocity": body_velocity,
                    "dodge_direction": direction
                }

            # 檢測蹲閃
            torso_length = abs(neck[1] - hip[1])
            expected_torso_length = shoulder_width * 1.4

            if torso_length < expected_torso_length * 0.8 and body_velocity > 0.3:
                confidence = min(0.6 + (1.0 - torso_length / expected_torso_length) * 0.3, 1.0)

                return {
                    "defense_type": BoxingDefenseType.DUCK.value,
                    "guard_hand": None,
                    "confidence": confidence,
                    "effectiveness": 0.85,
                    "coverage_areas": ["頭部蹲閃"],
                    "stamina_cost": 0.2,
                    "counter_ready": True,
                    "velocity": body_velocity,
                    "dodge_direction": "down"
                }

        except (IndexError, TypeError):
            pass

        return None

    def _calculate_defense_confidence(self, keypoint_conf, right_angle, left_angle,
                                        config: BoxingGuardConfig, velocity, velocity_bonus) -> float:
        """計算防禦置信度"""
        base_confidence = keypoint_conf * 0.3

        # 角度匹配度
        angle_score = 0.0
        valid_angles = 0

        if right_angle is not None:
            if self._angle_in_range(right_angle, config):
                angle_score += 1.0
            valid_angles += 1

        if left_angle is not None:
            if self._angle_in_range(left_angle, config):
                angle_score += 1.0
            valid_angles += 1

        if valid_angles > 0:
            angle_score /= valid_angles

        # 速度加分
        velocity_score = min(velocity * velocity_bonus, 0.2)

        # 總置信度
        total_confidence = base_confidence + angle_score * 0.5 + velocity_score
        return min(max(total_confidence, 0.0), 1.0)

    def _angle_in_range(self, angle, config: BoxingGuardConfig) -> bool:
        """檢查角度是否在配置範圍內"""
        if angle is None:
            return False
        return config.angle_min <= angle <= config.angle_max

    def _apply_defense_result(self, defense_data: PlayerDefenseData, defense_result: Dict):
        """應用防禦檢測結果到防禦數據"""
        defense_data.defense_type = defense_result["defense_type"]
        defense_data.guard_hand = defense_result.get("guard_hand")
        defense_data.confidence = defense_result["confidence"]
        defense_data.is_defending = True
        defense_data.velocity = defense_result.get("velocity", 0.0)
        defense_data.defense_effectiveness = defense_result["effectiveness"]
        defense_data.coverage_areas = defense_result["coverage_areas"]
        defense_data.stamina_cost = defense_result["stamina_cost"]
        defense_data.counter_ready = defense_result.get("counter_ready", False)

        # 處理閃避相關
        if "dodge_direction" in defense_result:
            defense_data.is_dodging = True
            defense_data.dodge_direction = defense_result["dodge_direction"]

    def _create_idle_defense_data(self, person_id: int, keypoints, timestamp: float) -> PlayerDefenseData:
        """創建閒置狀態的防禦數據"""
        try:
            person_keypoints = keypoints[person_id]

            right_arm_angle = calculate_normalized_angle(
                keypoints, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], person_id)
            left_arm_angle = calculate_normalized_angle(
                keypoints, [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], person_id)

            shoulder_width = calculate_shoulder_width(keypoints, person_id)
            body_center = self._get_body_center(person_keypoints)
            stance_stability = self._calculate_stance_stability(person_id, keypoints)

            return PlayerDefenseData(
                player_id=person_id,
                defense_type=BoxingDefenseType.IDLE.value,
                guard_hand=None,
                velocity=0.0,
                confidence=0.0,
                arm_angles={"right": right_arm_angle, "left": left_arm_angle},
                shoulder_width=shoulder_width,
                body_center=body_center,
                is_defending=False,
                is_dodging=False,
                dodge_direction=None,
                defense_effectiveness=0.0,
                coverage_areas=[],
                stamina_cost=0.0,
                counter_ready=False,
                stance_stability=stance_stability,
                guard_position={},
                timestamp=timestamp
            )
        except Exception as e:
            print(f"Error creating idle defense data for player {person_id}: {e}")
            return None

    def _calculate_stance_stability(self, person_id: int, keypoints) -> float:
        """計算站位穩定度"""
        try:
            person_keypoints = keypoints[person_id]

            # 獲取腳部關鍵點
            left_ankle = person_keypoints[LEFT_ANKLE]
            right_ankle = person_keypoints[RIGHT_ANKLE]
            left_knee = person_keypoints[LEFT_KNEE]
            right_knee = person_keypoints[RIGHT_KNEE]
            hip = person_keypoints[MID_HIP]

            # 檢查關鍵點置信度
            key_points = [left_ankle, right_ankle, left_knee, right_knee, hip]
            if any(point[2] < 0.3 for point in key_points):
                return 0.5  # 預設中等穩定度

            # 計算腳部距離（站位寬度）
            foot_distance = abs(left_ankle[0] - right_ankle[0])
            shoulder_width = calculate_shoulder_width(keypoints, person_id)

            if shoulder_width is None or shoulder_width == 0:
                return 0.5

            # 理想的站位寬度約為肩寬的0.8-1.2倍
            ideal_ratio = foot_distance / shoulder_width

            if 0.8 <= ideal_ratio <= 1.2:
                width_score = 1.0
            elif 0.6 <= ideal_ratio <= 1.4:
                width_score = 0.7
            else:
                width_score = 0.4

            # 檢查膝蓋是否在腳踝上方（垂直對齊）
            left_alignment = abs(left_knee[0] - left_ankle[0]) / shoulder_width
            right_alignment = abs(right_knee[0] - right_ankle[0]) / shoulder_width
            alignment_score = max(0, 1.0 - (left_alignment + right_alignment) / 2)

            # 綜合穩定度評分
            stability = (width_score * 0.6 + alignment_score * 0.4)
            return min(max(stability, 0.0), 1.0)

        except (IndexError, TypeError, ZeroDivisionError):
            return 0.5

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

    def _update_position_history(self, person_id: int, keypoints):
        """更新位置歷史記錄"""
        if person_id not in self.position_history:
            self.position_history[person_id] = deque(maxlen=self.history_length)

        try:
            person_keypoints = keypoints[person_id]

            # 記錄身體中心位置
            body_center = self._get_body_center(person_keypoints)
            if body_center:
                timestamp = time.time()
                self.position_history[person_id].append({
                    'center': body_center,
                    'timestamp': timestamp,
                    'neck': person_keypoints[NECK][:2] if person_keypoints[NECK][2] > 0.3 else None,
                     'hip': person_keypoints[MID_HIP][:2] if person_keypoints[MID_HIP][2] > 0.3 else None
                })

        except (IndexError, TypeError):
            pass

    def _calculate_arm_velocity(self, person_id: int, arm: str,
                                 current_positions: List[Tuple[float, float]]) -> float:
        """計算手臂速度"""
        if person_id not in self.position_history:
            return 0.0

        history = self.position_history[person_id]
        if len(history) < 2:
            return 0.0

        # 使用最近的位置計算速度
        recent_positions = list(history)[-3:]  # 最近3幀
        if len(recent_positions) < 2:
            return 0.0

        total_distance = 0.0
        total_time = 0.0

        for i in range(1, len(recent_positions)):
            prev = recent_positions[i - 1]['center']
            curr = recent_positions[i]['center']

            distance = math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
            time_diff = recent_positions[i]['timestamp'] - recent_positions[i - 1]['timestamp']

            total_distance += distance
            total_time += max(time_diff, 1e-3)  # 避免除零

        velocity = total_distance / total_time if total_time > 0 else 0.0
        return min(velocity, 5.0)  # 限制最大速度

    def _calculate_body_velocity(self, person_id: int, keypoints) -> float:
        """計算身體運動速度"""
        if person_id not in self.position_history:
            return 0.0

        history = self.position_history[person_id]
        if len(history) < 2:
            return 0.0

        recent_positions = list(history)[-2:]  # 最近2幀

        prev_center = recent_positions[0]['center']
        curr_center = recent_positions[1]['center']
        time_diff = recent_positions[1]['timestamp'] - recent_positions[0]['timestamp']

        if time_diff <= 0:
            return 0.0

        distance = math.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                                (curr_center[1] - prev_center[1]) ** 2)

        velocity = distance / time_diff
        return min(velocity, 3.0)  # 限制最大速度

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

# 防禦可視化器
class BoxingDefenseVisualizer:
    """拳擊防禦動作可視化器"""

    def __init__(self, show_skeleton=True, show_debug=True):
        self.show_skeleton = show_skeleton
        self.show_debug = show_debug
        self.last_frame_time = time.time()

    def draw_defense_frame(self, frame, keypoints, frame_data: DefenseFrameData):
        """繪製防禦信息幀"""
        if not self.show_debug:
            return frame

        result_frame = frame.copy()
        height, width = frame.shape[:2]

        # 繪製骨架
        if self.show_skeleton and keypoints is not None:
            for person_id in range(min(len(keypoints), 2)):
                self._draw_skeleton(result_frame, keypoints[person_id], person_id)

        # 繪製防禦信息
        for player_data in frame_data.players:
            self._draw_defense_info(result_frame, keypoints, player_data)

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

    def _draw_defense_info(self, frame, keypoints, player_data: PlayerDefenseData):
        """繪製防禦信息"""
        if keypoints is None or player_data.player_id >= len(keypoints):
            return

        person_keypoints = keypoints[player_data.player_id]
        color = (0, 255, 255) if player_data.player_id == 0 else (255, 255, 0)

        # 獲取頭部位置
        head_pos = None
        if person_keypoints[NOSE][2] > 0.3:
            head_pos = (int(person_keypoints[NOSE][0]), int(person_keypoints[NOSE][1]) - 50)
        elif person_keypoints[NECK][2] > 0.3:
            head_pos = (int(person_keypoints[NECK][0]), int(person_keypoints[NECK][1]) - 30)

        if head_pos:
            # 顯示防禦類型
            defense_text = player_data.defense_type.replace('_', ' ').title()
            cv2.putText(frame, defense_text, head_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 顯示防禦手
            if player_data.is_defending and player_data.guard_hand:
                hand_text = f"{player_data.guard_hand.upper()} GUARD"
                cv2.putText(frame, hand_text,
                            (head_pos[0], head_pos[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 顯示置信度
            if player_data.confidence > 0:
                conf_text = f"Conf: {player_data.confidence:.2f}"
                cv2.putText(frame, conf_text,
                            (head_pos[0], head_pos[1] + 45),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 顯示閃避方向
            if player_data.is_dodging and player_data.dodge_direction:
                dodge_text = f"DODGE {player_data.dodge_direction.upper()}"
                cv2.putText(frame, dodge_text,
                            (head_pos[0], head_pos[1] + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _draw_system_status(self, frame, frame_data: DefenseFrameData, width, height):
        """繪製系統狀態信息"""
        # 背景框
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 120), (255, 255, 255), 2)

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

        # 防禦統計
        defending_count = sum(1 for p in frame_data.players if p.is_defending)
        cv2.putText(frame, f"Defending: {defending_count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS計算
        current_time = time.time()
        fps = 1.0 / max(current_time - self.last_frame_time, 0.001)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.last_frame_time = current_time

def run_standalone_defense_detection(camera_index=0):
    """獨立運行防禦檢測（用於測試和調試）"""
    print("=== Running Boxing Defense Detection in Standalone Mode ===")

    detector = BoxingDefenseDetector()
    visualizer = BoxingDefenseVisualizer()

    try:
        for frame_id, keypoints_data, frame in get_keypoints_stream(video_source=camera_index):
            # 鏡像翻轉
            frame = cv2.flip(frame, 1)

            # 檢測防禦動作
            frame_data = detector.detect_defense_actions(keypoints_data)

            # 可視化結果
            result_frame = visualizer.draw_defense_frame(frame, keypoints_data, frame_data)

            # 打印檢測到的防禦動作
            for player in frame_data.players:
                if player.is_defending:
                    print(f"Player {player.player_id}: {player.defense_type} "
                            f"(conf: {player.confidence:.2f}, effectiveness: {player.defense_effectiveness:.2f})")

            # 顯示結果
            cv2.imshow('Boxing Defense Detection - Standalone', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Defense detection stopped by user")
    finally:
        cv2.destroyAllWindows()
        print("Defense detection completed")

if __name__ == "__main__":
    """主函數 - 僅用於測試防禦檢測"""
    import argparse

    parser = argparse.ArgumentParser(description='Boxing Defense Detection - Standalone Mode')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')

    args = parser.parse_args()

    # 運行獨立檢測模式
    run_standalone_defense_detection(args.camera)
