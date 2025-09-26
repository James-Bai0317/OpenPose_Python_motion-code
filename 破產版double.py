# dual_boxing_detection.py - 專注於雙人對戰動作辨識，可檢測攻擊、防禦及互動關係
import math
import time
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
import csv

# 引入原有模組
from pose_capture.openpose_api import get_keypoints_stream

# 引入角度計算模組
try:
    from angle import calculate_normalized_angle, calculate_shoulder_width
    from angle import (NOSE, NECK, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
                       LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, MID_HIP,
                       RIGHT_HIP, LEFT_HIP, RIGHT_KNEE, LEFT_KNEE,
                       RIGHT_ANKLE, LEFT_ANKLE)

    print("Successfully imported angle module functions")
except ImportError:
    print("Warning: angle.py not found, using built-in functions")

    # 內建關鍵點索引
    NOSE = 0;
    NECK = 1;
    RIGHT_SHOULDER = 2;
    RIGHT_ELBOW = 3;
    RIGHT_WRIST = 4
    LEFT_SHOULDER = 5;
    LEFT_ELBOW = 6;
    LEFT_WRIST = 7;
    MID_HIP = 8
    RIGHT_HIP = 9;
    LEFT_HIP = 10;
    RIGHT_KNEE = 11;
    LEFT_KNEE = 12
    RIGHT_ANKLE = 13;
    LEFT_ANKLE = 14


    def calculate_normalized_angle(keypoints, joint_indices, person_index):
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

# 引入防禦檢測模組
try:
    from guard_detection import BoxingDefenseDetector, BoxingDefenseVisualizer

    print("Successfully imported guard detection modules")
except ImportError:
    print("Warning: guard_detection.py not found, using basic defense detection")


    class BoxingDefenseDetector:
        def __init__(self):
            pass

        def detect_defense_actions(self, keypoints):
            return type('DefenseData', (), {'players': []})()


    class BoxingDefenseVisualizer:
        def __init__(self, show_skeleton=True, show_debug=True):
            pass


# 定義雙人對戰特有的動作類型
class DualActionType(Enum):
    IDLE = "idle"
    ATTACKING = "attacking"
    DEFENDING = "defending"
    DODGING = "dodging"
    COUNTER_ATTACKING = "counter_attacking"
    CLINCHING = "clinching"
    CIRCLING = "circling"


class CombatPhase(Enum):
    PREPARATION = "preparation"
    ENGAGEMENT = "engagement"
    EXCHANGE = "exchange"
    SEPARATION = "separation"
    REST = "rest"


class InteractionType(Enum):
    NO_INTERACTION = "no_interaction"
    FACING_OFF = "facing_off"
    ATTACKING_DEFENDING = "attacking_defending"
    MUTUAL_ATTACK = "mutual_attack"
    CLINCH = "clinch"
    CIRCLING = "circling"


# 雙人互動數據結構
@dataclass
class PlayerDualData:
    player_id: int
    action_type: str
    attack_hand: Optional[str]
    punch_type: Optional[str]
    defense_type: Optional[str]
    velocity: float
    confidence: float
    arm_angles: Dict[str, Optional[float]]
    shoulder_width: Optional[float]
    body_center: Optional[Tuple[float, float]]
    facing_direction: Optional[float]  # 面向角度
    stance_type: Optional[str]  # 站姿類型
    is_attacking: bool
    is_defending: bool
    is_dodging: bool
    target_player_id: Optional[int]  # 目標對手ID
    distance_to_opponent: Optional[float]
    relative_position: Optional[str]  # 相對位置 "left", "right", "front"
    combat_readiness: float  # 戰鬥準備度 0-1
    timestamp: float


@dataclass
class CombatInteraction:
    interaction_type: str
    aggressor_id: Optional[int]
    defender_id: Optional[int]
    attack_type: Optional[str]
    defense_type: Optional[str]
    distance: float
    relative_angle: float
    success_likelihood: float  # 攻擊成功可能性
    counter_opportunity: bool  # 是否有反擊機會
    timestamp: float


@dataclass
class DualFrameData:
    frame_id: int
    timestamp: float
    players: List[PlayerDualData]
    combat_phase: str
    interaction: Optional[CombatInteraction]
    players_distance: Optional[float]
    combat_intensity: float  # 戰鬥激烈程度 0-1
    ring_center: Optional[Tuple[float, float]]  # 擂台中心


class DualBoxingDetector:
    """雙人拳擊對戰檢測器"""

    def __init__(self):
        self.frame_count = 0

        # 拳法配置 (繼承原有配置)
        self.punch_configs = {
            "straight": {"name": "直拳", "angle_threshold": 120, "speed_threshold": 0.02, "min_extension": 70},
            "hook": {"name": "勾拳", "angle_threshold": 80, "speed_threshold": 0.03, "min_extension": 70},
            "uppercut": {"name": "上勾拳", "angle_threshold": 50, "speed_threshold": 0.02, "min_extension": 70}
        }

        # 防禦檢測器
        try:
            self.defense_detector = BoxingDefenseDetector()
        except:
            self.defense_detector = None

        # 雙人對戰特有參數
        self.combat_distance_threshold = 250  # 戰鬥距離閾值(像素)
        self.close_combat_distance = 150  # 近戰距離
        self.engagement_distance = 300  # 交戰距離

        # 歷史記錄
        self.position_history = {}
        self.combat_history = deque(maxlen=30)  # 30幀戰鬥歷史
        self.interaction_history = deque(maxlen=15)  # 15幀互動歷史

        # 冷卻機制
        self.player_cooldowns = {}
        self.cooldown_frames = 8

        print("=== Dual Boxing Detector Initialized ===")

    def detect_dual_actions(self, keypoints) -> DualFrameData:
        """檢測雙人對戰動作"""
        self.frame_count += 1
        timestamp = time.time()

        players_data = []
        combat_phase = CombatPhase.PREPARATION.value
        interaction = None
        players_distance = None
        combat_intensity = 0.0
        ring_center = None

        if keypoints is not None and len(keypoints) >= 2:
            # 確保處理兩個人
            keypoints = keypoints[:2]  # 只取前兩個人

            # 計算基本距離和位置信息
            players_distance = self._calculate_players_distance(keypoints)
            ring_center = self._calculate_ring_center(keypoints)

            # 檢測每個玩家的動作
            for person_id in range(2):
                player_data = self._detect_dual_player_action(person_id, keypoints, timestamp)
                if player_data:
                    # 添加對戰特有信息
                    self._add_dual_context(player_data, keypoints, person_id, players_distance)
                    players_data.append(player_data)

                # 更新位置歷史
                self._update_position_history(person_id, keypoints)

            # 分析雙人互動
            if len(players_data) == 2:
                interaction = self._analyze_combat_interaction(players_data, keypoints)
                combat_phase = self._determine_combat_phase(players_data, interaction, players_distance)
                combat_intensity = self._calculate_combat_intensity(players_data, interaction)

        # 創建幀數據
        frame_data = DualFrameData(
            frame_id=self.frame_count,
            timestamp=timestamp,
            players=players_data,
            combat_phase=combat_phase,
            interaction=interaction,
            players_distance=players_distance,
            combat_intensity=combat_intensity,
            ring_center=ring_center
        )

        # 更新歷史
        self.combat_history.append(frame_data)
        if interaction:
            self.interaction_history.append(interaction)

        return frame_data

    def _detect_dual_player_action(self, person_id: int, keypoints, timestamp: float) -> Optional[PlayerDualData]:
        """檢測雙人模式下的單個玩家動作"""
        try:
            person_keypoints = keypoints[person_id]

            # 檢查冷卻
            if person_id in self.player_cooldowns:
                if self.frame_count - self.player_cooldowns[person_id] < self.cooldown_frames:
                    return None

            # 計算基本角度
            right_arm_angle = calculate_normalized_angle(
                keypoints, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], person_id)
            left_arm_angle = calculate_normalized_angle(
                keypoints, [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], person_id)

            shoulder_width = calculate_shoulder_width(keypoints, person_id)
            body_center = self._get_body_center(person_keypoints)
            facing_direction = self._calculate_facing_direction(person_keypoints)
            stance_type = self._detect_stance_type(person_keypoints, right_arm_angle, left_arm_angle)
            combat_readiness = self._calculate_combat_readiness(person_keypoints, right_arm_angle, left_arm_angle)

            # 初始化玩家數據
            player_data = PlayerDualData(
                player_id=person_id,
                action_type=DualActionType.IDLE.value,
                attack_hand=None,
                punch_type=None,
                defense_type=None,
                velocity=0.0,
                confidence=0.0,
                arm_angles={"right": right_arm_angle, "left": left_arm_angle},
                shoulder_width=shoulder_width,
                body_center=body_center,
                facing_direction=facing_direction,
                stance_type=stance_type,
                is_attacking=False,
                is_defending=False,
                is_dodging=False,
                target_player_id=1 - person_id,  # 對手ID
                distance_to_opponent=None,
                relative_position=None,
                combat_readiness=combat_readiness,
                timestamp=timestamp
            )

            # 檢測防禦動作
            if self._detect_guard_pose(person_id, keypoints, right_arm_angle, left_arm_angle):
                player_data.action_type = DualActionType.DEFENDING.value
                player_data.is_defending = True
                player_data.defense_type = "guard"
                self.player_cooldowns[person_id] = self.frame_count

            # 檢測閃避動作
            elif self._detect_dodge_action(person_id, keypoints):
                player_data.action_type = DualActionType.DODGING.value
                player_data.is_dodging = True
                self.player_cooldowns[person_id] = self.frame_count

            # 檢測攻擊動作
            else:
                punch_result = self._detect_punch_action(person_id, keypoints, right_arm_angle, left_arm_angle)
                if punch_result:
                    player_data.action_type = DualActionType.ATTACKING.value
                    player_data.attack_hand = punch_result["hand"]
                    player_data.punch_type = punch_result["punch_type"]
                    player_data.velocity = punch_result["velocity"]
                    player_data.confidence = punch_result["confidence"]
                    player_data.is_attacking = True
                    self.player_cooldowns[person_id] = self.frame_count

            return player_data

        except Exception as e:
            print(f"Error detecting dual player {person_id} action: {e}")
            return None

    def _add_dual_context(self, player_data: PlayerDualData, keypoints, person_id: int,
                          players_distance: Optional[float]):
        """添加雙人對戰上下文信息"""
        try:
            opponent_id = 1 - person_id
            if opponent_id < len(keypoints):
                # 計算相對位置
                player_center = player_data.body_center
                opponent_center = self._get_body_center(keypoints[opponent_id])

                if player_center and opponent_center:
                    # 相對位置
                    if player_center[0] < opponent_center[0]:
                        player_data.relative_position = "left"
                    elif player_center[0] > opponent_center[0]:
                        player_data.relative_position = "right"
                    else:
                        player_data.relative_position = "center"

                    # 到對手的距離
                    player_data.distance_to_opponent = math.sqrt(
                        (player_center[0] - opponent_center[0]) ** 2 +
                        (player_center[1] - opponent_center[1]) ** 2
                    )
        except Exception as e:
            print(f"Error adding dual context: {e}")

    def _analyze_combat_interaction(self, players_data: List[PlayerDualData], keypoints) -> Optional[CombatInteraction]:
        """分析戰鬥互動"""
        if len(players_data) != 2:
            return None

        try:
            player0, player1 = players_data[0], players_data[1]
            distance = player0.distance_to_opponent or 0

            # 計算相對角度
            relative_angle = 0.0
            if player0.body_center and player1.body_center:
                dx = player1.body_center[0] - player0.body_center[0]
                dy = player1.body_center[1] - player0.body_center[1]
                relative_angle = math.degrees(math.atan2(dy, dx))

            # 確定互動類型
            interaction_type = InteractionType.NO_INTERACTION.value
            aggressor_id = None
            defender_id = None
            attack_type = None
            defense_type = None
            success_likelihood = 0.0
            counter_opportunity = False

            # 分析攻防關係
            if player0.is_attacking and player1.is_defending:
                interaction_type = InteractionType.ATTACKING_DEFENDING.value
                aggressor_id = 0
                defender_id = 1
                attack_type = player0.punch_type
                defense_type = player1.defense_type
                success_likelihood = self._calculate_attack_success(player0, player1, distance)
                counter_opportunity = self._has_counter_opportunity(player1, player0)

            elif player1.is_attacking and player0.is_defending:
                interaction_type = InteractionType.ATTACKING_DEFENDING.value
                aggressor_id = 1
                defender_id = 0
                attack_type = player1.punch_type
                defense_type = player0.defense_type
                success_likelihood = self._calculate_attack_success(player1, player0, distance)
                counter_opportunity = self._has_counter_opportunity(player0, player1)

            elif player0.is_attacking and player1.is_attacking:
                interaction_type = InteractionType.MUTUAL_ATTACK.value
                # 互相攻擊時，選擇速度較快的作為主攻擊者
                if player0.velocity > player1.velocity:
                    aggressor_id = 0
                    attack_type = player0.punch_type
                else:
                    aggressor_id = 1
                    attack_type = player1.punch_type
                success_likelihood = 0.3  # 互相攻擊成功率較低

            elif distance < self.close_combat_distance:
                if not player0.is_attacking and not player1.is_attacking:
                    interaction_type = InteractionType.CLINCH.value
                else:
                    interaction_type = InteractionType.FACING_OFF.value

            elif distance < self.engagement_distance:
                interaction_type = InteractionType.FACING_OFF.value

            return CombatInteraction(
                interaction_type=interaction_type,
                aggressor_id=aggressor_id,
                defender_id=defender_id,
                attack_type=attack_type,
                defense_type=defense_type,
                distance=distance,
                relative_angle=relative_angle,
                success_likelihood=success_likelihood,
                counter_opportunity=counter_opportunity,
                timestamp=time.time()
            )

        except Exception as e:
            print(f"Error analyzing combat interaction: {e}")
            return None

    def _calculate_attack_success(self, attacker: PlayerDualData, defender: PlayerDualData, distance: float) -> float:
        """計算攻擊成功可能性"""
        success = 0.0

        # 距離因素
        optimal_distance = 180  # 最佳攻擊距離
        if distance <= optimal_distance:
            distance_factor = 1.0 - abs(distance - optimal_distance) / optimal_distance
        else:
            distance_factor = max(0, 1.0 - (distance - optimal_distance) / 100)

        success += distance_factor * 0.3

        # 攻擊者信心度
        success += attacker.confidence * 0.4

        # 防守者準備度 (越高，攻擊成功率越低)
        if defender.is_defending:
            success -= defender.combat_readiness * 0.3
        else:
            success += 0.2  # 對手沒有防守

        # 攻擊速度
        if attacker.velocity > 1.0:
            success += 0.1

        return max(0, min(success, 1.0))

    def _has_counter_opportunity(self, defender: PlayerDualData, attacker: PlayerDualData) -> bool:
        """判斷是否有反擊機會"""
        if not defender.is_defending:
            return False

        # 如果攻擊者置信度低，防守者有反擊機會
        if attacker.confidence < 0.6:
            return True

        # 如果防守者戰鬥準備度高，有反擊機會
        if defender.combat_readiness > 0.7:
            return True

        return False

    def _determine_combat_phase(self, players_data: List[PlayerDualData],
                                interaction: Optional[CombatInteraction],
                                distance: Optional[float]) -> str:
        """確定戰鬥階段"""
        if not players_data or len(players_data) != 2:
            return CombatPhase.PREPARATION.value

        if distance is None:
            return CombatPhase.PREPARATION.value

        # 根據距離和動作確定階段
        if distance > self.engagement_distance:
            return CombatPhase.PREPARATION.value

        elif distance > self.combat_distance_threshold:
            if any(p.is_attacking for p in players_data):
                return CombatPhase.ENGAGEMENT.value
            else:
                return CombatPhase.PREPARATION.value

        elif distance > self.close_combat_distance:
            if interaction and interaction.interaction_type in [
                InteractionType.ATTACKING_DEFENDING.value,
                InteractionType.MUTUAL_ATTACK.value
            ]:
                return CombatPhase.EXCHANGE.value
            else:
                return CombatPhase.ENGAGEMENT.value

        else:  # 非常近距離
            if any(p.is_attacking for p in players_data):
                return CombatPhase.EXCHANGE.value
            else:
                return CombatPhase.REST.value

    def _calculate_combat_intensity(self, players_data: List[PlayerDualData],
                                    interaction: Optional[CombatInteraction]) -> float:
        """計算戰鬥激烈程度"""
        intensity = 0.0

        # 基於玩家動作
        for player in players_data:
            if player.is_attacking:
                intensity += 0.4 * player.confidence
            if player.is_defending:
                intensity += 0.2 * player.combat_readiness
            intensity += player.velocity * 0.1

        # 基於互動類型
        if interaction:
            if interaction.interaction_type == InteractionType.MUTUAL_ATTACK.value:
                intensity += 0.3
            elif interaction.interaction_type == InteractionType.ATTACKING_DEFENDING.value:
                intensity += 0.2
            elif interaction.interaction_type == InteractionType.CLINCH.value:
                intensity += 0.1

        return min(intensity, 1.0)

    # 以下方法繼承並改進原有功能
    def _detect_guard_pose(self, person_id: int, keypoints, right_arm_angle, left_arm_angle) -> bool:
        """檢測防禦姿勢 - 雙人版本"""
        if right_arm_angle is None or left_arm_angle is None:
            return False

        guard_angle_min, guard_angle_max = 40, 140
        right_guard = guard_angle_min <= right_arm_angle <= guard_angle_max
        left_guard = guard_angle_min <= left_arm_angle <= guard_angle_max

        if not (right_guard and left_guard):
            return False

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

    def _detect_dodge_action(self, person_id: int, keypoints) -> bool:
        """檢測閃避動作"""
        try:
            if person_id not in self.position_history:
                return False

            history = self.position_history[person_id]
            if len(history.get('body', [])) < 5:
                return False

            recent_positions = history['body'][-5:]

            # 檢查垂直移動 (下蹲閃避)
            y_positions = [pos[1] for pos in recent_positions]
            max_y, min_y = max(y_positions), min(y_positions)

            if max_y - min_y > 40:  # 垂直移動超過40像素
                return True

            # 檢查水平快速移動 (側閃)
            x_positions = [pos[0] for pos in recent_positions]
            x_movement = abs(x_positions[-1] - x_positions[0])

            if x_movement > 60:  # 水平移動超過60像素
                return True

            return False

        except Exception:
            return False

    def _detect_punch_action(self, person_id: int, keypoints, right_arm_angle, left_arm_angle) -> Optional[Dict]:
        """檢測拳擊動作 - 改進的雙人版本"""
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
        """分析拳擊動作類型 - 雙人優化版"""
        if arm_angle is None:
            return None

        try:
            person_keypoints = keypoints[person_id]

            # 選擇手臂關鍵點
            if hand == "right":
                wrist_idx, elbow_idx, shoulder_idx = RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER
            else:
                wrist_idx, elbow_idx, shoulder_idx = LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER

            wrist = person_keypoints[wrist_idx]
            elbow = person_keypoints[elbow_idx]
            shoulder = person_keypoints[shoulder_idx]

            # 置信度檢查
            if wrist[2] < 0.3 or elbow[2] < 0.3 or shoulder[2] < 0.3:
                return None

            # 計算手臂伸展
            wrist_x_offset = wrist[0] - shoulder[0]
            wrist_y_offset = wrist[1] - shoulder[1]
            arm_extension = math.sqrt(wrist_x_offset ** 2 + wrist_y_offset ** 2)

            # 計算速度
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

            # 檢測每種拳法
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
                        "action_type": DualActionType.ATTACKING.value,
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

    def _calculate_ring_center(self, keypoints) -> Optional[Tuple[float, float]]:
        """計算擂台中心點"""
        try:
            if len(keypoints) < 2:
                return None

            center_0 = self._get_body_center(keypoints[0])
            center_1 = self._get_body_center(keypoints[1])

            if center_0 is None or center_1 is None:
                return None

            ring_center = ((center_0[0] + center_1[0]) / 2, (center_0[1] + center_1[1]) / 2)
            return ring_center

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

    def _calculate_facing_direction(self, person_keypoints) -> Optional[float]:
        """計算面向方向"""
        try:
            left_shoulder = person_keypoints[LEFT_SHOULDER]
            right_shoulder = person_keypoints[RIGHT_SHOULDER]

            if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
                # 計算肩膀向量的垂直方向作為面向
                shoulder_vector = (right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1])
                facing_angle = math.degrees(math.atan2(-shoulder_vector[0], shoulder_vector[1]))
                return facing_angle

        except (IndexError, TypeError, ZeroDivisionError):
            pass

        return None

    def _detect_stance_type(self, person_keypoints, right_arm_angle: Optional[float],
                            left_arm_angle: Optional[float]) -> Optional[str]:
        """檢測站姿類型"""
        try:
            # 基於腳部位置判斷站姿
            left_ankle = person_keypoints[LEFT_ANKLE]
            right_ankle = person_keypoints[RIGHT_ANKLE]

            if left_ankle[2] > 0.3 and right_ankle[2] > 0.3:
                foot_distance = abs(left_ankle[0] - right_ankle[0])

                if foot_distance > 100:
                    # 基於前腳判斷正架或反架
                    if left_ankle[1] < right_ankle[1]:  # 左腳在前
                        return "orthodox"  # 正架
                    else:
                        return "southpaw"  # 反架
                else:
                    return "square"  # 平行站姿

        except (IndexError, TypeError):
            pass

        return "unknown"

    def _calculate_combat_readiness(self, person_keypoints, right_arm_angle: Optional[float],
                                    left_arm_angle: Optional[float]) -> float:
        """計算戰鬥準備度"""
        readiness = 0.0

        try:
            # 基於手臂位置
            if right_arm_angle is not None and left_arm_angle is not None:
                # 手臂彎曲度 (越彎曲，準備度越高)
                avg_bend = (180 - right_arm_angle + 180 - left_arm_angle) / 2
                readiness += min(avg_bend / 120, 0.4)

            # 基於身體姿態
            neck = person_keypoints[NECK]
            hip = person_keypoints[MID_HIP]

            if neck[2] > 0.3 and hip[2] > 0.3:
                # 身體前傾度
                body_lean = abs(neck[0] - hip[0])
                readiness += min(body_lean / 50, 0.3)

            # 基於腳步寬度
            left_ankle = person_keypoints[LEFT_ANKLE]
            right_ankle = person_keypoints[RIGHT_ANKLE]

            if left_ankle[2] > 0.3 and right_ankle[2] > 0.3:
                stance_width = abs(left_ankle[0] - right_ankle[0])
                if stance_width > 80:  # 有戰鬥站姿
                    readiness += 0.3

        except (IndexError, TypeError):
            pass

        return min(readiness, 1.0)

    def _calculate_hand_velocity(self, person_id: int, hand: str, current_pos: Tuple[float, float]) -> float:
        """計算手部速度"""
        if person_id not in self.position_history:
            return 0.0

        if hand not in self.position_history[person_id]:
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
        return min(velocity, 5.0)  # 限制最大速度

    def _update_position_history(self, person_id: int, keypoints):
        """更新位置歷史記錄"""
        if person_id not in self.position_history:
            self.position_history[person_id] = {
                "right": [], "left": [], "body": []
            }

        try:
            person_keypoints = keypoints[person_id]
            current_time = time.time()

            # 更新手腕位置
            for hand, wrist_idx in [("right", RIGHT_WRIST), ("left", LEFT_WRIST)]:
                wrist = person_keypoints[wrist_idx]
                if wrist[2] > 0.3:
                    history = self.position_history[person_id][hand]
                    history.append((wrist[0], wrist[1], current_time))

                    if len(history) > 8:
                        history.pop(0)

            # 更新身體中心位置
            body_center = self._get_body_center(person_keypoints)
            if body_center:
                body_history = self.position_history[person_id]["body"]
                body_history.append((body_center[0], body_center[1], current_time))

                if len(body_history) > 10:
                    body_history.pop(0)

        except (IndexError, TypeError):
            pass


class DualBoxingVisualizer:
    """雙人拳擊可視化器"""

    def __init__(self, show_skeleton=True, show_debug=True):
        self.show_skeleton = show_skeleton
        self.show_debug = show_debug
        self.last_frame_time = time.time()

    def draw_dual_frame(self, frame, keypoints, frame_data: DualFrameData):
        """繪製雙人對戰幀"""
        if not self.show_debug:
            return frame

        result_frame = frame.copy()
        height, width = frame.shape[:2]

        # 繪製骨架
        if self.show_skeleton and keypoints is not None:
            for person_id in range(min(len(keypoints), 2)):
                self._draw_skeleton(result_frame, keypoints[person_id], person_id)

        # 繪製玩家動作信息
        for player_data in frame_data.players:
            self._draw_player_info(result_frame, keypoints, player_data)

        # 繪製對戰信息
        self._draw_combat_info(result_frame, frame_data, width, height)

        # 繪製互動關係
        if frame_data.interaction and len(frame_data.players) == 2:
            self._draw_interaction(result_frame, keypoints, frame_data.interaction, frame_data.players)

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

    def _draw_player_info(self, frame, keypoints, player_data: PlayerDualData):
        """繪製玩家信息"""
        if keypoints is None or player_data.player_id >= len(keypoints):
            return

        person_keypoints = keypoints[player_data.player_id]
        color = (0, 255, 0) if player_data.player_id == 0 else (255, 0, 0)

        # 獲取頭部位置
        head_pos = self._get_head_position(person_keypoints)
        if not head_pos:
            return

        y_offset = 0

        # 顯示動作類型
        action_text = player_data.action_type.replace('_', ' ').title()
        cv2.putText(frame, action_text, (head_pos[0], head_pos[1] + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 25

        # 顯示攻擊信息
        if player_data.is_attacking:
            attack_text = f"{player_data.attack_hand.upper()} {player_data.punch_type.upper()}"
            cv2.putText(frame, attack_text, (head_pos[0], head_pos[1] + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 20

            conf_text = f"Conf: {player_data.confidence:.2f}"
            cv2.putText(frame, conf_text, (head_pos[0], head_pos[1] + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_offset += 20

        # 顯示防禦信息
        if player_data.is_defending:
            defense_text = f"DEFENSE: {player_data.defense_type.upper()}"
            cv2.putText(frame, defense_text, (head_pos[0], head_pos[1] + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 20

        # 顯示戰鬥準備度
        readiness_text = f"Ready: {player_data.combat_readiness:.2f}"
        cv2.putText(frame, readiness_text, (head_pos[0], head_pos[1] + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 20

        # 顯示站姿
        if player_data.stance_type:
            stance_text = f"Stance: {player_data.stance_type.upper()}"
            cv2.putText(frame, stance_text, (head_pos[0], head_pos[1] + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _draw_combat_info(self, frame, frame_data: DualFrameData, width, height):
        """繪製對戰信息"""
        # 背景框
        info_height = 180
        cv2.rectangle(frame, (10, 10), (400, info_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, info_height), (255, 255, 255), 2)

        y_pos = 30
        line_height = 20

        # 基本信息
        cv2.putText(frame, f"Frame: {frame_data.frame_id}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height

        cv2.putText(frame, f"Players: {len(frame_data.players)}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height

        # 戰鬥階段
        cv2.putText(frame, f"Phase: {frame_data.combat_phase.upper()}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += line_height

        # 玩家距離
        if frame_data.players_distance is not None:
            cv2.putText(frame, f"Distance: {frame_data.players_distance:.1f}px", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_height

        # 戰鬥激烈程度
        intensity_color = (0, 255, 0) if frame_data.combat_intensity < 0.5 else (0, 255,
                                                                                 255) if frame_data.combat_intensity < 0.8 else (
            0, 0, 255)
        cv2.putText(frame, f"Intensity: {frame_data.combat_intensity:.2f}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, intensity_color, 2)
        y_pos += line_height

        # 互動類型
        if frame_data.interaction:
            interaction_text = f"Interaction: {frame_data.interaction.interaction_type.replace('_', ' ').upper()}"
            cv2.putText(frame, interaction_text, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y_pos += line_height

        # FPS
        current_time = time.time()
        fps = 1.0 / max(current_time - self.last_frame_time, 0.001)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.last_frame_time = current_time

    def _draw_interaction(self, frame, keypoints, interaction: CombatInteraction, players: List[PlayerDualData]):
        """繪製互動關係"""
        if len(players) != 2 or not keypoints or len(keypoints) < 2:
            return

        player0_center = players[0].body_center
        player1_center = players[1].body_center

        if not player0_center or not player1_center:
            return

        # 繪製連線
        color = (255, 255, 255)
        if interaction.interaction_type == InteractionType.ATTACKING_DEFENDING.value:
            color = (0, 0, 255)  # 紅色表示攻防
        elif interaction.interaction_type == InteractionType.MUTUAL_ATTACK.value:
            color = (0, 255, 255)  # 黃色表示互攻
        elif interaction.interaction_type == InteractionType.CLINCH.value:
            color = (255, 0, 255)  # 紫色表示貼身

        cv2.line(frame,
                 (int(player0_center[0]), int(player0_center[1])),
                 (int(player1_center[0]), int(player1_center[1])),
                 color, 3)

        # 在連線中點顯示互動信息
        mid_x = int((player0_center[0] + player1_center[0]) / 2)
        mid_y = int((player0_center[1] + player1_center[1]) / 2)

        # 顯示攻擊成功率
        if interaction.success_likelihood > 0:
            success_text = f"Hit: {interaction.success_likelihood:.1%}"
            cv2.putText(frame, success_text, (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 顯示反擊機會
        if interaction.counter_opportunity:
            cv2.putText(frame, "COUNTER!", (mid_x, mid_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _get_head_position(self, person_keypoints):
        """獲取頭部位置"""
        if person_keypoints[NOSE][2] > 0.3:
            return (int(person_keypoints[NOSE][0]), int(person_keypoints[NOSE][1]) - 50)
        elif person_keypoints[NECK][2] > 0.3:
            return (int(person_keypoints[NECK][0]), int(person_keypoints[NECK][1]) - 30)
        return None


def run_dual_boxing_detection(camera_index=0, ground_truth_action="dual_test"):
    """運行雙人拳擊對戰檢測"""
    print("=== Running Dual Boxing Combat Detection ===")
    print(f"=== Testing: {ground_truth_action} ===")

    detector = DualBoxingDetector()
    visualizer = DualBoxingVisualizer()

    # 日誌設定
    log_filename = f"dual_log_{ground_truth_action}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    log_file = open(log_filename, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)

    # CSV標頭
    header = [
        "timestamp", "frame_id", "player1_action", "player1_confidence",
        "player2_action", "player2_confidence", "combat_phase", "interaction_type",
        "players_distance", "combat_intensity", "success_likelihood"
    ]
    log_writer.writerow(header)

    try:
        for frame_id, keypoints_data, frame in get_keypoints_stream(video_source=camera_index):
            # 檢測雙人對戰
            frame_data = detector.detect_dual_actions(keypoints_data)

            # 寫入日誌
            if len(frame_data.players) >= 2:
                player1, player2 = frame_data.players[0], frame_data.players[1]
                log_row = [
                    time.time(), frame_data.frame_id,
                    player1.action_type, player1.confidence,
                    player2.action_type, player2.confidence,
                    frame_data.combat_phase,
                    frame_data.interaction.interaction_type if frame_data.interaction else "none",
                    frame_data.players_distance or 0,
                    frame_data.combat_intensity,
                    frame_data.interaction.success_likelihood if frame_data.interaction else 0
                ]
                log_writer.writerow(log_row)

            # 可視化
            result_frame = visualizer.draw_dual_frame(frame, keypoints_data, frame_data)

            # 終端輸出
            if len(frame_data.players) >= 2:
                p1, p2 = frame_data.players[0], frame_data.players[1]
                print(f"P1: {p1.action_type} | P2: {p2.action_type} | "
                      f"Phase: {frame_data.combat_phase} | "
                      f"Intensity: {frame_data.combat_intensity:.2f}")

                if frame_data.interaction:
                    print(f"  -> Interaction: {frame_data.interaction.interaction_type}")

            cv2.imshow('Dual Boxing Combat Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Detection stopped by user")
    finally:
        if log_file:
            log_file.close()
            print(f"Dual combat log saved to: {log_filename}")
        cv2.destroyAllWindows()
        print("Dual boxing detection completed")


if __name__ == "__main__":
    """主函數 - 雙人拳擊對戰檢測"""
    import argparse

    parser = argparse.ArgumentParser(description='Dual Boxing Combat Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--action', type=str, default="dual_combat_test",
                        help='Ground truth action for dual combat test',nargs='+')

    args = parser.parse_args()

    # 運行雙人對戰檢測
    run_dual_boxing_detection(camera_index=args.camera, ground_truth_action=args.action)
