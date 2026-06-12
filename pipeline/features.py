import librosa
import numpy as np


N_MFCC = 13


def extract_mfcc(waveform: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    MFCC 특징을 추출합니다.

    MFCC는 발음의 전체적인 음색/조음 차이를 숫자 벡터로 표현합니다.
    MVP에서는 시간축 전체를 평균내서 13차원 벡터로 사용합니다.

    Args:
        waveform: 오디오 파형 데이터
        sr: 샘플링 레이트
        n_mfcc: 추출할 MFCC 개수

    Returns:
        mfcc_mean: shape = (13,)
    """
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)

    # mfcc shape: (n_mfcc, frames)
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


def extract_zcr(waveform: np.ndarray) -> float:
    """
    Zero Crossing Rate를 추출합니다.

    ZCR은 파형이 0을 얼마나 자주 가로지르는지 나타냅니다.
    /θ/, /f/ 같은 마찰음은 고주파 노이즈가 많아서 ZCR이 비교적 높게 나옵니다.

    Args:
        waveform: 오디오 파형 데이터

    Returns:
        zcr_mean: 평균 ZCR 값
    """
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    zcr = librosa.feature.zero_crossing_rate(waveform)

    return float(np.mean(zcr))


def extract_duration_ms(waveform: np.ndarray, sr: int) -> float:
    """
    오디오 길이를 ms 단위로 계산합니다.

    주의:
        MVP에서는 음소 길이가 아니라 단어 전체 길이를 사용합니다.
        /iː/ vs /i/처럼 장단 차이가 큰 경우 단어 길이를 프록시로 활용합니다.

    Args:
        waveform: 오디오 파형 데이터
        sr: 샘플링 레이트

    Returns:
        duration_ms: 오디오 길이(ms)
    """
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    return float(len(waveform) / sr * 1000)


def extract_rms(waveform: np.ndarray) -> float:
    """
    RMS energy를 추출합니다.

    RMS는 음성의 평균 에너지 크기를 나타냅니다.
    /ə/가 너무 강하게 발음되는지 보는 데 사용할 수 있습니다.

    Args:
        waveform: 오디오 파형 데이터

    Returns:
        rms_mean: 평균 RMS 값
    """
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    rms = librosa.feature.rms(y=waveform)

    return float(np.mean(rms))


def extract_spectral_centroid(waveform: np.ndarray, sr: int) -> float:
    """
    Spectral Centroid를 추출합니다.

    Spectral Centroid는 소리의 무게중심이 어느 주파수대에 있는지 나타냅니다.
    값이 높을수록 더 밝고 날카로운 소리에 가깝습니다.

    Args:
        waveform: 오디오 파형 데이터
        sr: 샘플링 레이트

    Returns:
        centroid_mean: 평균 spectral centroid 값
    """
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)

    return float(np.mean(centroid))


def extract_features(waveform: np.ndarray, sr: int) -> dict[str, float | list[float]]:
    """
    채점에 필요한 모든 음향 특징을 한 번에 추출합니다.

    Args:
        waveform: VAD 묵음 제거가 끝난 오디오 파형 데이터
        sr: 샘플링 레이트

    Returns:
        features: 채점과 reference vector 생성에 사용할 특징 dict
    """
    mfcc_mean = extract_mfcc(waveform, sr)

    return {
        # JSON 저장을 쉽게 하기 위해 numpy array를 list로 변환합니다.
        "mfcc_mean": mfcc_mean.tolist(),
        "zcr_mean": extract_zcr(waveform),
        "duration_ms": extract_duration_ms(waveform, sr),
        "rms_mean": extract_rms(waveform),
        "spectral_centroid_mean": extract_spectral_centroid(waveform, sr),
    }
