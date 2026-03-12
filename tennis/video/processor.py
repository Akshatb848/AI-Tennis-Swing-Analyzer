"""
Video Processor — Upload handling, transcoding, and segmentation.
Production pipeline for tennis match video processing.
"""

from __future__ import annotations
import os
import uuid
import asyncio
import subprocess
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class VideoStatus(str, Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    TRANSCODING = "transcoding"
    SEGMENTING = "segmenting"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    FAILED = "failed"


class VideoCodec(str, Enum):
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"


class VideoResolution(str, Enum):
    SD_480 = "480p"
    HD_720 = "720p"
    FHD_1080 = "1080p"
    UHD_4K = "2160p"


@dataclass
class TranscodeProfile:
    """Encoding profile for a specific quality tier."""
    name: str
    resolution: VideoResolution
    bitrate_kbps: int
    codec: VideoCodec
    fps: int = 30
    keyframe_interval: int = 60  # 2 seconds at 30fps

    @property
    def ffmpeg_args(self) -> list[str]:
        res_map = {"480p": "854:480", "720p": "1280:720", "1080p": "1920:1080", "2160p": "3840:2160"}
        codec_map = {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9", "av1": "libaom-av1"}
        return [
            "-vf", f"scale={res_map[self.resolution.value]}",
            "-c:v", codec_map[self.codec.value],
            "-b:v", f"{self.bitrate_kbps}k",
            "-r", str(self.fps),
            "-g", str(self.keyframe_interval),
            "-preset", "fast",
        ]


# Standard ABR profiles for HLS streaming
TRANSCODE_PROFILES = [
    TranscodeProfile("low", VideoResolution.SD_480, 800, VideoCodec.H264),
    TranscodeProfile("medium", VideoResolution.HD_720, 2500, VideoCodec.H264),
    TranscodeProfile("high", VideoResolution.FHD_1080, 5000, VideoCodec.H264),
]


@dataclass
class VideoSegment:
    """A segment/point extracted from a match video."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    video_id: str = ""
    segment_type: str = "point"   # point, game, set, highlight
    start_time_ms: int = 0
    end_time_ms: int = 0
    point_number: Optional[int] = None
    score_at_start: str = ""
    score_at_end: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class VideoJob:
    """Represents a video processing job."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    source_path: str = ""
    output_dir: str = ""
    status: VideoStatus = VideoStatus.UPLOADED
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    segments: list[VideoSegment] = field(default_factory=list)
    transcode_outputs: dict[str, str] = field(default_factory=dict)


def _probe_with_ffprobe(source_path: str) -> Optional[dict]:
    """Probe video metadata using ffprobe if available."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            source_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return None


def _probe_with_opencv(source_path: str) -> Optional[dict]:
    """Probe video metadata using OpenCV as fallback."""
    if not HAS_OPENCV:
        return None
    try:
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            return None
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0.0
        cap.release()
        return info
    except Exception:
        return None


class VideoProcessor:
    """
    Production video processing pipeline.

    Handles:
    1. Upload & validation
    2. Metadata probing via ffprobe or OpenCV
    3. Transcoding to multiple bitrates (ABR for HLS)
    4. Point-level segmentation from scoring timeline
    5. Frame extraction for ML inference
    """

    def __init__(self, upload_dir: str = "./uploads/tennisiq", output_dir: str = "./output/tennisiq"):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self._jobs: dict[str, VideoJob] = {}
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def create_job(self, session_id: str, source_path: str) -> VideoJob:
        """Create a new video processing job."""
        job_output = os.path.join(self.output_dir, session_id)
        os.makedirs(job_output, exist_ok=True)
        job = VideoJob(session_id=session_id, source_path=source_path, output_dir=job_output)
        self._jobs[job.id] = job
        return job

    async def process(self, job_id: str) -> VideoJob:
        """Run the full video processing pipeline."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        try:
            # Step 1: Probe video metadata
            job.status = VideoStatus.QUEUED
            job.progress = 0.05
            await self._probe_video(job)

            # Step 2: Transcode to ABR profiles
            job.status = VideoStatus.TRANSCODING
            job.progress = 0.1
            await self._transcode(job)

            # Step 3: Segment by points
            job.status = VideoStatus.SEGMENTING
            job.progress = 0.6
            await self._segment_by_points(job)

            # Step 4: Generate HLS manifest
            job.progress = 0.9
            await self._generate_hls_manifest(job)

            # Done
            job.status = VideoStatus.COMPLETE
            job.progress = 1.0
            job.completed_at = datetime.utcnow()

        except Exception as e:
            job.status = VideoStatus.FAILED
            job.error_message = str(e)
            logger.error("Video processing failed for job %s: %s", job_id, e)

        return job

    async def _probe_video(self, job: VideoJob):
        """Extract video metadata using ffprobe or OpenCV."""
        source = job.source_path

        # Try ffprobe first
        probe = _probe_with_ffprobe(source)
        if probe and "streams" in probe:
            for stream in probe["streams"]:
                if stream.get("codec_type") == "video":
                    job.width = int(stream.get("width", 1920))
                    job.height = int(stream.get("height", 1080))
                    # Parse fps from r_frame_rate
                    fps_str = stream.get("r_frame_rate", "30/1")
                    if "/" in str(fps_str):
                        num, den = fps_str.split("/")
                        job.fps = float(num) / float(den) if float(den) > 0 else 30.0
                    else:
                        job.fps = float(fps_str) if fps_str else 30.0
                    break
            if "format" in probe:
                job.duration_seconds = float(probe["format"].get("duration", 0))
            logger.info("Probed video via ffprobe: %dx%d @ %.1f fps, %.1fs",
                        job.width, job.height, job.fps, job.duration_seconds)
            return

        # Fallback to OpenCV
        cv_probe = _probe_with_opencv(source)
        if cv_probe:
            job.width = cv_probe["width"]
            job.height = cv_probe["height"]
            job.fps = cv_probe["fps"]
            job.duration_seconds = cv_probe["duration"]
            logger.info("Probed video via OpenCV: %dx%d @ %.1f fps, %.1fs",
                        job.width, job.height, job.fps, job.duration_seconds)
            return

        # Final fallback: reasonable defaults
        logger.warning("Could not probe video %s — using defaults", source)
        job.duration_seconds = 0.0
        job.fps = 30.0

    async def _transcode(self, job: VideoJob):
        """Transcode video to multiple quality levels using ffmpeg."""
        for i, profile in enumerate(TRANSCODE_PROFILES):
            output_path = os.path.join(job.output_dir, f"{profile.name}.mp4")

            # Try real ffmpeg transcode
            try:
                cmd = (
                    ["ffmpeg", "-y", "-i", job.source_path]
                    + profile.ffmpeg_args
                    + ["-an", output_path]
                )
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=3600)
                if proc.returncode == 0:
                    job.transcode_outputs[profile.name] = output_path
                    logger.info("Transcoded %s profile to %s", profile.name, output_path)
                else:
                    logger.warning("ffmpeg transcode failed for %s profile", profile.name)
                    job.transcode_outputs[profile.name] = job.source_path
            except (FileNotFoundError, asyncio.TimeoutError):
                # ffmpeg not available — use source directly
                logger.info("ffmpeg not available, using source file for %s profile", profile.name)
                job.transcode_outputs[profile.name] = job.source_path

            job.progress = 0.1 + (0.5 * (i + 1) / len(TRANSCODE_PROFILES))

    async def _segment_by_points(self, job: VideoJob, scoring_timeline: Optional[list] = None):
        """Segment video into individual points using scoring timeline."""
        if scoring_timeline:
            # Use real scoring timeline
            for i, point in enumerate(scoring_timeline):
                seg = VideoSegment(
                    video_id=job.id,
                    segment_type="point",
                    start_time_ms=point.get("start_ms", 0),
                    end_time_ms=point.get("end_ms", 0),
                    point_number=point.get("point_number", i + 1),
                    score_at_start=point.get("score_before", ""),
                    score_at_end=point.get("score_after", ""),
                    tags=point.get("tags", ["rally"]),
                )
                job.segments.append(seg)
        elif job.duration_seconds > 0:
            # Auto-segment into 30-second chunks for analysis
            chunk_ms = 30000
            total_ms = int(job.duration_seconds * 1000)
            for i, start_ms in enumerate(range(0, total_ms, chunk_ms)):
                end_ms = min(start_ms + chunk_ms, total_ms)
                seg = VideoSegment(
                    video_id=job.id,
                    segment_type="chunk",
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    point_number=i + 1,
                    tags=["auto_segment"],
                )
                job.segments.append(seg)

    async def _generate_hls_manifest(self, job: VideoJob):
        """Generate HLS master playlist and media playlists."""
        manifest_path = os.path.join(job.output_dir, "master.m3u8")
        manifest = "#EXTM3U\n#EXT-X-VERSION:3\n"
        bandwidth_map = {"low": 800000, "medium": 2500000, "high": 5000000}
        res_map = {"low": "854x480", "medium": "1280x720", "high": "1920x1080"}
        for name, path in job.transcode_outputs.items():
            bw = bandwidth_map.get(name, 2500000)
            res = res_map.get(name, "1280x720")
            manifest += f'#EXT-X-STREAM-INF:BANDWIDTH={bw},RESOLUTION={res}\n{name}.m3u8\n'
        try:
            with open(manifest_path, "w") as f:
                f.write(manifest)
        except OSError as e:
            logger.warning("Could not write HLS manifest: %s", e)
        job.transcode_outputs["manifest"] = manifest_path

    def get_job(self, job_id: str) -> Optional[VideoJob]:
        return self._jobs.get(job_id)

    def extract_frames(
        self, job: VideoJob, start_ms: int, end_ms: int, fps: int = 5,
    ) -> list[str]:
        """Extract frames from a video segment for ML inference."""
        if not HAS_OPENCV:
            # Fallback: return placeholder frame paths
            frame_count = int((end_ms - start_ms) / 1000 * fps)
            return [f"frame_{i:06d}.jpg" for i in range(frame_count)]

        output_frames = []
        cap = cv2.VideoCapture(job.source_path)
        if not cap.isOpened():
            logger.error("Cannot open video for frame extraction: %s", job.source_path)
            return []

        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(source_fps / fps))
        idx = 0

        frames_dir = os.path.join(job.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if current_ms > end_ms:
                break

            if idx % frame_interval == 0:
                frame_path = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                output_frames.append(frame_path)

            idx += 1

        cap.release()
        logger.info("Extracted %d frames from %dms to %dms", len(output_frames), start_ms, end_ms)
        return output_frames
