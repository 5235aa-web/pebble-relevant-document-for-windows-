"""
IndexTTS2 引擎模块 - 用于 Windows 平台的文字转语音

该模块为 Conscious Pebble AI 聊天机器人提供本地 TTS 功能
针对 RTX 3050 (4GB VRAM) 进行了优化

特性:
- 速度-质量平衡: fast/balanced/quality 三种模式
- FP16 推理: 减少显存占用约 40-50%
- 音频缓存: 相同文本直接返回缓存文件
- 情感控制: 支持情感向量和文本情感
- 云端支持: 支持 ElevenLabs 云端 TTS

使用方式:
    from tts_engine import TTSEngine

    tts = TTSEngine(quality_preset="balanced")  # balanced 是默认值
    audio = tts.synthesize("你好，我是AI助手")
"""

import hashlib
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tools import get_voice_config

# Try to import cloud TTS
try:
    from cloud_tts import synthesize_voice_bytes as cloud_synthesize
    CLOUD_TTS_AVAILABLE = True
except ImportError:
    CLOUD_TTS_AVAILABLE = False
    cloud_synthesize = None
import inspect
import os
import tempfile
from typing import Optional

import torch
import numpy as np
import soundfile as sf


class TTSEngine:
    """
    IndexTTS2 文字转语音引擎

    特性:
    - 懒加载: 仅在首次使用时加载模型，节省 VRAM
    - FP16 推理: 减少显存占用约 40-50%
    - 音频缓存: 相同文本直接返回缓存文件
    - 情感控制: 支持情感向量和文本情感
    - 速度-质量平衡: fast/balanced/quality 三种模式
    """

    def __init__(
        self,
        model_dir: str = "checkpoints",
        device: str = "cuda",
        cache_dir: str = "data/tts_cache",
        use_fp16: bool = True,
        quality_preset: str = "balanced"
    ):
        """
        初始化 TTS 引擎

        Args:
            model_dir: 模型文件目录 (默认: "checkpoints")
            device: 运行设备 (默认: "cuda", 支持 "cpu")
            cache_dir: 音频缓存目录
            use_fp16: 是否使用 FP16 推理 (减少显存占用)
            quality_preset: 速度-质量平衡 ("fast", "balanced", "quality")
        """
        self.model = None
        self.model_dir = Path(model_dir)
        # 强制使用 cuda 如果可用
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_fp16 = use_fp16 and (self.device == "cuda")

        # 速度-质量平衡设置
        self.quality_preset = quality_preset
        self._apply_quality_preset()

        # 音频参数
        self.sample_rate = 24000  # IndexTTS2 默认采样率

        # 默认参考音频 (用于语音克隆)
        self.default_spk_audio = None

        print(f"[TTS] IndexTTS2 引擎初始化完成")
        print(f"[TTS] 设备: {self.device}")
        print(f"[TTS] FP16: {'开启' if self.use_fp16 else '关闭'}")
        print(f"[TTS] 质量模式: {self.quality_preset}")
        print(f"[TTS] 模型目录: {self.model_dir}")

    def _apply_quality_preset(self):
        """应用速度-质量平衡设置"""
        presets = {
            "fast": {
                "max_len": 100,        # 减少生成步数
                "spk_layers": 2,       # 减少说话人层数
                "temperature": 0.8,     # 稍低的温度
            },
            "balanced": {
                "max_len": 200,        # 平衡设置
                "spk_layers": 4,
                "temperature": 1.0,
            },
            "quality": {
                "max_len": 500,        # 最大生成步数
                "spk_layers": 6,       # 更多说话人层数
                "temperature": 1.0,
            }
        }

        if self.quality_preset not in presets:
            print(f"[TTS] 警告: 未知质量模式 '{self.quality_preset}'，使用 'balanced'")
            self.quality_preset = "balanced"

        self.preset_settings = presets[self.quality_preset]
        print(f"[TTS] 预设参数: {self.preset_settings}")

    def set_quality_preset(self, preset: str):
        """
        动态更改质量模式

        Args:
            preset: "fast", "balanced", 或 "quality"
        """
        if preset != self.quality_preset:
            self.quality_preset = preset
            self._apply_quality_preset()
            print(f"[TTS] 质量模式已切换为: {preset}")

    def _find_reference_audio(self) -> Optional[str]:
        """查找默认的参考音频"""
        # 先查找 examples 文件夹
        base_dir = Path(__file__).parent
        examples_dir = base_dir / "index-tts" / "examples"

        if examples_dir.exists():
            wav_files = list(examples_dir.glob("*.wav"))
            if wav_files:
                return str(wav_files[0])

        # 如果没有，创建一个简单的参考音频
        return None

    def _load_model(self):
        """懒加载模型 - 仅在首次使用时加载"""
        if self.model is not None:
            return

        print("[TTS] 正在加载 IndexTTS2 模型...")

        try:
            # 动态导入 IndexTTS2
            try:
                from indextts.infer_v2 import IndexTTS2
            except ImportError:
                # 尝试添加 index-tts 目录到路径
                index_tts_path = Path(__file__).parent / "index-tts"
                if index_tts_path.exists():
                    sys.path.insert(0, str(index_tts_path))
                    from indextts.infer_v2 import IndexTTS2
                else:
                    raise ImportError("index-tts 目录未找到")

            # 创建 IndexTTS2 实例 - 启用 CUDA 核心加速
            self.model = IndexTTS2(
                cfg_path=str(self.model_dir / "config.yaml"),
                model_dir=str(self.model_dir),
                use_fp16=self.use_fp16,
                use_cuda_kernel=True,  # 启用 CUDA 核心加速
                use_deepspeed=False
            )

            # 启用 PyTorch 推理优化
            self._setup_inference_optimizations()

            # 查找参考音频
            self.default_spk_audio = self._find_reference_audio()
            if self.default_spk_audio:
                print(f"[TTS] 使用参考音频: {self.default_spk_audio}")
            else:
                print("[TTS] 警告: 未找到参考音频，可能影响音质")

            print(f"[TTS] 模型加载成功!")

        except Exception as e:
            print(f"[TTS] 模型加载失败: {e}")
            raise

    def _setup_inference_optimizations(self):
        """设置推理优化"""
        if self.device == "cuda" and torch.cuda.is_available():
            # 启用 cuDNN 自动优化
            torch.backends.cudnn.benchmark = True

            # 启用 TF32 (Ampere+ GPU)
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            print("[TTS] CUDA 推理优化已启用")

    def _get_cache_path(self, cache_key: str) -> Optional[Path]:
        """获取文本对应的缓存文件路径"""
        text_hash = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
        cache_path = self.cache_dir / f"{text_hash}.wav"

        if cache_path.exists():
            return cache_path
        return None

    def _save_cache(self, cache_key: str, audio: np.ndarray, text: str = "", voice: str = "", speed: float = 1.0) -> Path:
        """保存音频到缓存"""
        text_hash = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
        cache_path = self.cache_dir / f"{text_hash}.wav"

        sf.write(cache_path, audio, self.sample_rate)
        return cache_path

    def synthesize(
        self,
        text: str,
        voice: str = "neutral",
        speed: float = 1.0,
        use_cache: bool = True,
        spk_audio: str = None,
        max_len: int = None
    ) -> Optional[bytes]:
        """
        将文本转换为语音

        Args:
            text: 要转换的文本
            voice: 语音风格 (neutral, happy, sad, angry, excited, calm)
            speed: 语速倍数 (0.5 - 2.0)
            use_cache: 是否使用缓存
            spk_audio: 参考音频路径 (可选)
            max_len: 最大声谱帧数 (覆盖预设值)

        Returns:
            WAV 格式的音频字节数据, 失败返回 None
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # 确定实际使用的 max_len
        effective_max_len = max_len if max_len is not None else self.preset_settings.get("max_len", 200)

        # 检查缓存 (包含 max_len 以区分不同质量设置)
        if use_cache:
            cache_key = f"{text}_{voice}_{speed}_{effective_max_len}"
            cache_path = self._get_cache_path(cache_key)
            if cache_path:
                print(f"[TTS] 使用缓存: {text[:20]}...")
                with open(cache_path, 'rb') as f:
                    return f.read()

        # 确保模型已加载
        try:
            self._load_model()
        except Exception as e:
            print(f"[TTS] 无法加载模型: {e}")
            return None

        print(f"[TTS] 合成语音 [{self.quality_preset}模式, max_len={effective_max_len}]: {text[:30]}...")

        try:
            # 使用参考音频
            audio_prompt = spk_audio or self.default_spk_audio

            if not audio_prompt:
                print("[TTS] 错误: 需要参考音频来进行语音合成")
                return None

            # 准备输出文件
            import uuid
            output_path = str(self.cache_dir / f"temp_{uuid.uuid4().hex}.wav")

            # 设置情感参数
            emo_vector = self._get_emotion_vector(voice)

            # 调用 IndexTTS2 推理 - 传递优化参数
            # 尝试传递 max_len 参数（如果模型支持）
            infer_kwargs = {
                "spk_audio_prompt": audio_prompt,
                "text": text,
                "output_path": output_path,
                "emo_vector": emo_vector,
                "use_random": False,
                "verbose": False
            }

            # 如果模型支持 max_len 参数，传递它
            if hasattr(self.model, 'max_len') or 'max_len' in inspect.signature(self.model.infer).parameters:
                infer_kwargs["max_len"] = effective_max_len

            self.model.infer(**infer_kwargs)

            # 读取生成的音频
            audio, _ = sf.read(output_path, dtype='float32')

            # 确保音频是正确格式 (mono)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # 如果是立体声，转为单声道
            elif audio.ndim == 0:
                audio = audio.reshape(-1)  # 确保是一维数组

            # 保存到缓存
            cache_path = self._save_cache(cache_key, audio, text, voice, speed)
            print(f"[TTS] 语音已保存: {cache_path}")

            # 清理临时文件
            if Path(output_path).exists():
                Path(output_path).unlink()

            # 读取为字节返回
            with open(cache_path, 'rb') as f:
                return f.read()

        except Exception as e:
            print(f"[TTS] 合成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_emotion_vector(self, voice: str) -> list:
        """
        获取情感向量
        情感顺序: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        """
        emotion_map = {
            "neutral": [0, 0, 0, 0, 0, 0, 0, 0.3],
            "happy": [0.7, 0, 0, 0, 0, 0, 0.3, 0],
            "sad": [0, 0, 0.7, 0, 0, 0.3, 0, 0],
            "angry": [0, 0.7, 0, 0.3, 0.3, 0, 0, 0],
            "excited": [0.8, 0, 0, 0, 0, 0, 0.5, 0],
            "calm": [0, 0, 0, 0, 0, 0, 0, 0.6],
            "fear": [0, 0.3, 0.3, 0.7, 0.3, 0.3, 0.2, 0],
        }
        return emotion_map.get(voice, [0, 0, 0, 0, 0, 0, 0, 0])

    def unload(self):
        """卸载模型以释放显存"""
        if self.model is not None:
            del self.model
            self.model = None

            if self.device == "cuda":
                torch.cuda.empty_cache()

            print("[TTS] 模型已卸载，显存已释放")

    def cleanup_cache(self, max_size_mb: int = 500):
        """
        清理旧缓存文件

        Args:
            max_size_mb: 最大缓存大小 (MB)
        """
        total_size = 0
        files = sorted(
            self.cache_dir.glob("*.wav"),
            key=lambda x: x.stat().st_mtime
        )

        for f in files:
            total_size += f.stat().st_size / (1024 * 1024)

        # 如果超过最大 size，删除最老的文件
        while total_size > max_size_mb and files:
            oldest = files.pop(0)
            oldest.unlink()
            total_size -= oldest.stat().st_size / (1024 * 1024)
            print(f"[TTS] 已删除缓存: {oldest.name}")


# 全局 TTS 实例
_tts_engine: Optional[TTSEngine] = None


def get_tts_engine(quality_preset: str = "balanced") -> TTSEngine:
    """
    获取全局 TTS 引擎实例

    Args:
        quality_preset: 速度-质量平衡 ("fast", "balanced", "quality")
    """
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine(quality_preset=quality_preset)
    return _tts_engine


def synthesize_voice_bytes(
    text: str,
    voice_name: str = "neutral",
    detected_emotion: str = None,
    quality_preset: str = "balanced",
    speed: float = 1.0,
    spk_audio: str = None,
    force_provider: str = None
) -> Optional[bytes]:
    """
    兼容旧接口的语音合成函数

    Args:
        text: 要转换的文本
        voice_name: 语音名称
        detected_emotion: 检测到的情感
        quality_preset: 速度-质量平衡 ("fast", "balanced", "quality")
        speed: 语速倍数
        spk_audio: 参考音频路径
        force_provider: 强制使用指定提供商 ("local", "elevenlabs")，可选

    Returns:
        WAV 格式的音频字节数据
    """
    # 映射语音名称到情感
    emotion_map = {
        "Pebble": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "excited": "excited",
        "calm": "calm",
    }

    # 如果传入了检测到的情感，优先使用
    voice = emotion_map.get(voice_name, "neutral")
    if detected_emotion and detected_emotion in emotion_map.values():
        voice = detected_emotion

    # 获取配置决定使用哪个 TTS 提供商
    config = get_voice_config()
    tts_provider = force_provider or config.get("tts_provider", "local")

    # 使用云端 TTS
    if tts_provider == "elevenlabs" and CLOUD_TTS_AVAILABLE:
        api_key = config.get("elevenlabs_api_key", "")
        # Default to official example voice ID
        voice_id = config.get("elevenlabs_voice_id", "JBFqnCBsd6RMkjVDRZzb")

        if not api_key:
            print("[TTS] ElevenLabs API key not configured")
            return None
        else:
            print(f"[TTS] Using ElevenLabs TTS (voice: {voice_id})")
            # Cloud TTS returns MP3 data, save to temp file and return path
            mp3_data = cloud_synthesize(
                text=text,
                voice_id=voice_id,
                api_key=api_key,
                speed=speed,
                model_id="eleven_multilingual_v2"
            )
            if mp3_data:
                import tempfile
                import uuid
                # Save MP3 to temp file with .mp3 extension
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                temp_file.write(mp3_data)
                temp_file.close()
                print(f"[TTS] Cloud TTS saved to: {temp_file.name}")
                return temp_file.name
            else:
                print("[TTS] Cloud TTS failed, not using local fallback")
                return None

    # Only use local TTS when explicitly set to local
    engine = get_tts_engine(quality_preset=quality_preset)
    return engine.synthesize(
        text,
        voice=voice,
        speed=speed,
        spk_audio=spk_audio
    )


def unload_tts():
    """卸载 TTS 模型释放显存"""
    global _tts_engine
    if _tts_engine:
        _tts_engine.unload()
        _tts_engine = None
