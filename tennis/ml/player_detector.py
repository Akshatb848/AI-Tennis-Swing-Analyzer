"""
Player Detector — Player detection, pose estimation, and ID tracking.
Wraps PlayerNet model with DeepSORT-style tracking.
"""

from __future__ import annotations

from tennis.models.events import (
    BoundingBox,
    EventType,
    PlayerEvent,
    PlayerPose,
    Point2D,
    PoseKeypoint,
)

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class PlayerTrack:
    """Tracked player with ID persistence."""
    def __init__(self, player_id: str, bbox: BoundingBox):
        self.player_id = player_id
        self.bbox = bbox
        self.frames_seen = 1
        self.frames_missed = 0
        self.pose: PlayerPose | None = None
        self.court_position: Point2D | None = None

    @property
    def is_active(self) -> bool:
        return self.frames_missed < 30

    def update(self, bbox: BoundingBox, pose: PlayerPose | None = None):
        self.bbox = bbox
        self.frames_seen += 1
        self.frames_missed = 0
        if pose:
            self.pose = pose

    def mark_missed(self):
        self.frames_missed += 1


class PlayerDetector:
    """Player detection + pose estimation + ID tracking."""

    def __init__(self, max_players: int = 4):
        self.max_players = max_players
        self.tracks: dict[str, PlayerTrack] = {}
        self._next_id = 0

    def process_frame(
        self,
        detections: list[BoundingBox],
        keypoints_list: list[list[tuple[float, float, float]]] | None = None,
        frame_number: int = 0,
        session_id: str = "",
        fps: float = 30.0,
    ) -> list[PlayerEvent]:
        """Process detections for one frame. Returns PlayerEvents."""
        events = []

        # Match detections to existing tracks (simple IoU matching)
        matched_tracks = set()
        for i, det in enumerate(detections[:self.max_players]):
            best_track = self._match_detection(det)
            kps = keypoints_list[i] if keypoints_list and i < len(keypoints_list) else None
            pose = self._build_pose(kps) if kps else None

            if best_track:
                best_track.update(det, pose)
                matched_tracks.add(best_track.player_id)
                pid = best_track.player_id
            else:
                pid = f"player_{self._next_id}"
                self._next_id += 1
                track = PlayerTrack(pid, det)
                track.pose = pose
                self.tracks[pid] = track

            event = PlayerEvent(
                event_type=EventType.PLAYER_POSITION,
                timestamp_ms=int(frame_number / fps * 1000),
                frame_number=frame_number,
                session_id=session_id,
                player_id=pid,
                position_image=det,
                pose=pose,
                detection_confidence=det.confidence,
            )
            events.append(event)

        # Mark unmatched tracks
        for pid, track in self.tracks.items():
            if pid not in matched_tracks:
                track.mark_missed()

        # Remove dead tracks
        self.tracks = {pid: t for pid, t in self.tracks.items() if t.is_active}
        return events

    def _match_detection(self, det: BoundingBox) -> PlayerTrack | None:
        """Match detection to existing track via IoU."""
        best_iou = 0.3
        best_track = None
        for track in self.tracks.values():
            iou = self._calc_iou(det, track.bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        return best_track

    @staticmethod
    def _calc_iou(a: BoundingBox, b: BoundingBox) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _build_pose(keypoints: list[tuple[float, float, float]]) -> PlayerPose:
        kps = []
        for i, (x, y, conf) in enumerate(keypoints):
            name = COCO_KEYPOINT_NAMES[i] if i < len(COCO_KEYPOINT_NAMES) else f"kp_{i}"
            kps.append(PoseKeypoint(name=name, x=x, y=y, confidence=conf))
        overall = sum(k.confidence for k in kps) / len(kps) if kps else 0
        return PlayerPose(keypoints=kps, overall_confidence=overall)

    def get_active_players(self) -> list[str]:
        return [pid for pid, t in self.tracks.items() if t.is_active]

    def reset(self):
        self.tracks.clear()
        self._next_id = 0
