# Holosoma 프로젝트 완전 가이드

> **Holosoma** (그리스어: "whole-body", 전신) — Amazon FAR 팀이 개발한 **휴머노이드 로봇 강화학습 프레임워크**
>
> 시뮬레이션에서 정책(policy)을 학습하고, 실제 로봇에 배포(deploy)하는 전체 파이프라인을 제공한다.

---

## 목차

1. [프로젝트 한눈에 보기](#1-프로젝트-한눈에-보기)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [핵심 개념 정리](#3-핵심-개념-정리)
4. [세 개의 서브패키지 상세](#4-세-개의-서브패키지-상세)
   - 4.1 [holosoma (학습)](#41-holosoma--학습-핵심)
   - 4.2 [holosoma_inference (추론/배포)](#42-holosoma_inference--추론배포)
   - 4.3 [holosoma_retargeting (모션 리타겟팅)](#43-holosoma_retargeting--모션-리타겟팅)
5. [Manager 기반 아키텍처](#5-manager-기반-아키텍처)
6. [학습 루프 상세 흐름](#6-학습-루프-상세-흐름)
7. [설정(Config) 시스템](#7-설정config-시스템)
8. [지원 로봇/시뮬레이터/알고리즘](#8-지원-로봇시뮬레이터알고리즘)
9. [데이터 파이프라인: 모션 데이터 → 학습](#9-데이터-파이프라인-모션-데이터--학습)
10. [실행 방법 요약](#10-실행-방법-요약)
11. [내 데이터를 넣으려면?](#11-내-데이터를-넣으려면)

---

## 1. 프로젝트 한눈에 보기

Holosoma가 하는 일을 한 문장으로 요약하면:

> **사람의 모션 캡처 데이터를 로봇 모션으로 변환(retarget)하고, 시뮬레이터에서 강화학습(RL)으로 정책을 훈련한 뒤, 실제 로봇에 배포한다.**

```
┌──────────────────────────────────────────────────────────────┐
│                    전체 파이프라인                             │
│                                                              │
│  ① 모션 캡처 데이터      ② 리타겟팅          ③ 데이터 변환    │
│  (AMASS, LAFAN,   ──→  사람→로봇 모션  ──→  .npz 형식으로    │
│   OMOMO 등)              변환                 시뮬레이터용     │
│                                               변환            │
│          ④ 시뮬레이터에서 RL 학습                             │
│          (IsaacGym / IsaacSim / MJWarp)                      │
│                    │                                         │
│                    ▼                                         │
│          ⑤ ONNX 모델 내보내기                                │
│                    │                                         │
│                    ▼                                         │
│          ⑥ 실제 로봇 or MuJoCo에서 추론(inference)           │
└──────────────────────────────────────────────────────────────┘
```

### 두 가지 Task

| Task | 설명 | 예시 |
|------|------|------|
| **Locomotion** (보행) | 속도 명령(전진, 회전 등)을 추적하며 걷기 | "앞으로 0.5 m/s로 걸어라" |
| **Whole-Body Tracking (WBT)** | 미리 녹화된 모션 클립을 전신으로 따라하기 | "이 춤 동작을 따라해라" |

---

## 2. 디렉토리 구조

```
holosoma/
├── src/
│   ├── holosoma/                    # ★ 핵심: 학습 프레임워크
│   │   └── holosoma/
│   │       ├── train_agent.py       #   학습 진입점
│   │       ├── eval_agent.py        #   평가 & ONNX 내보내기
│   │       ├── replay.py            #   모션 재생
│   │       ├── agents/              #   RL 알고리즘 (PPO, FastSAC)
│   │       ├── envs/                #   Task 환경 (Locomotion, WBT)
│   │       ├── managers/            #   9개 매니저 (핵심 아키텍처)
│   │       ├── simulator/           #   시뮬레이터 추상화 계층
│   │       ├── config_types/        #   설정 타입 정의 (dataclass)
│   │       ├── config_values/       #   실제 설정 프리셋 값
│   │       ├── data/                #   로봇 URDF, 모션 데이터
│   │       ├── bridge/              #   실제 로봇 SDK 연결
│   │       └── utils/               #   유틸리티
│   │
│   ├── holosoma_inference/          # 추론 & 배포 파이프라인
│   │   └── holosoma_inference/
│   │       ├── run_policy.py        #   추론 진입점
│   │       ├── policies/            #   Locomotion, WBT 추론 정책
│   │       ├── models/              #   사전학습된 ONNX 모델
│   │       └── docs/workflows/      #   배포 가이드 문서
│   │
│   └── holosoma_retargeting/        # 모션 리타겟팅
│       └── holosoma_retargeting/
│           ├── examples/            #   리타겟팅 실행 스크립트
│           ├── data_conversion/     #   학습용 데이터 변환
│           ├── data_utils/          #   모캡 데이터 전처리
│           ├── src/                 #   리타겟팅 알고리즘 핵심
│           └── evaluation/          #   리타겟팅 품질 평가
│
├── scripts/                         # 환경 설정 셸 스크립트
├── demo_scripts/                    # 전체 파이프라인 데모
├── tests/                           # 테스트 (CI, E2E, Nightly)
├── docker/                          # Docker 설정
└── pyproject.toml                   # 프로젝트 메타데이터
```

---

## 3. 핵심 개념 정리

로보틱스 RL을 처음 접하는 입장에서 알아야 할 핵심 용어들:

### 강화학습(RL) 기초

| 용어 | 설명 |
|------|------|
| **Policy (정책)** | 관측(observation)을 받아 행동(action)을 출력하는 신경망 |
| **Observation** | 로봇이 현재 느끼는 감각 정보 (관절 각도, 속도, IMU, 명령 등) |
| **Action** | 정책이 출력하는 값 → 각 관절의 목표 위치(target position) |
| **Reward** | 잘했으면 +, 못했으면 -. 여러 항목의 가중합으로 계산 |
| **Episode** | 한 번의 시뮬레이션 에피소드 (로봇이 넘어지거나 시간 초과 시 리셋) |
| **Domain Randomization** | 시뮬레이션 환경을 무작위로 변화시켜 현실 적응력(sim-to-real)을 높이는 기법 |
| **Curriculum Learning** | 쉬운 것부터 점점 어렵게 학습 난이도를 올리는 기법 |

### 시뮬레이터 관련

| 용어 | 설명 |
|------|------|
| **IsaacGym** | NVIDIA의 GPU 가속 물리 시뮬레이터 (수천 개 환경 병렬 실행) |
| **IsaacSim** | NVIDIA Omniverse 기반 시뮬레이터 (렌더링 + 물리) |
| **MJWarp** | MuJoCo의 GPU 가속 버전 (베타) |
| **MuJoCo** | DeepMind의 물리 시뮬레이터 (추론 시 사용) |
| **URDF** | 로봇 모델 파일 형식 (링크, 관절, 질량 등 정의) |

### 모션 데이터 관련

| 용어 | 설명 |
|------|------|
| **Motion Retargeting** | 사람 모션 → 로봇 모션으로 변환하는 과정 |
| **SMPL-X / SMPL-H** | 사람 몸 형태를 파라미터로 표현하는 모델 |
| **AMASS** | 대규모 모션 캡처 데이터셋 |
| **LAFAN** | Ubisoft의 모션 캡처 데이터셋 |
| **OMOMO** | 사람-물체 상호작용 모션 데이터셋 |
| **.npz** | numpy 배열 묶음 파일, 모션 데이터의 기본 포맷 |

---

## 4. 세 개의 서브패키지 상세

### 4.1 `holosoma` — 학습 핵심

이 패키지가 전체의 핵심이다. RL 학습의 모든 요소를 담고 있다.

#### 진입점: `train_agent.py`

```python
# 실행 흐름 요약
def main():
    config = tyro.cli(...)           # CLI에서 설정 파싱
    train(config)

def train(config):
    init_sim_imports(config)         # 시뮬레이터 초기화
    env = get_class(env_target)(config)  # 환경 생성 (Locomotion 또는 WBT)
    algo = algo_class(env=env, ...)  # 알고리즘 생성 (PPO 또는 FastSAC)
    algo.setup()                     # 신경망, 옵티마이저 초기화
    algo.learn()                     # ← 여기서 수만 번 반복 학습
```

#### 핵심 클래스 관계도

```
BaseAlgo (agents/base_algo/)
  ├── PPO (agents/ppo/)
  └── FastSAC (agents/fast_sac/)
        │
        │  algo.learn() 내부에서 반복:
        │    obs = env.step(action)
        │    reward 계산 → 정책 업데이트
        │
        ▼
BaseTask (envs/base_task/)
  ├── LeggedRobotLocomotionManager (envs/locomotion/)  ← Locomotion Task
  └── WholeBodyTrackingManager (envs/wbt/)             ← WBT Task
        │
        │  내부에 9개 Manager 보유:
        │
        ├── ObservationManager   ← 무엇을 관측할지
        ├── ActionManager        ← 행동을 관절 명령으로 변환
        ├── RewardManager        ← 보상 계산
        ├── CommandManager       ← 명령 생성 (속도 or 모션 클립)
        ├── TerminationManager   ← 에피소드 종료 조건
        ├── RandomizationManager ← 도메인 랜덤화
        ├── CurriculumManager    ← 커리큘럼 학습
        ├── TerrainManager       ← 지형 생성
        └── ResetEventManager    ← 리셋 이벤트
              │
              ▼
        BaseSimulator (simulator/base_simulator/)
          ├── IsaacGymSimulator
          ├── IsaacSimSimulator
          └── MuJoCoSimulator
```

#### Agents (RL 알고리즘)

**PPO (Proximal Policy Optimization)**
- 가장 널리 쓰이는 on-policy RL 알고리즘
- `agents/ppo/` 에 구현
- Actor-Critic 구조: Actor(정책) + Critic(가치함수)

**FastSAC (Soft Actor-Critic 변형)**
- Off-policy 알고리즘. 샘플 효율이 더 좋음
- `agents/fast_sac/` 에 구현
- Distributional critic (C51 스타일) 사용

#### Environments (Task 환경)

**LeggedRobotLocomotionManager** — 속도 추적 보행
- 명령: 선속도(x, y), 각속도(yaw)
- 보상: 명령 추적 정확도, 에너지 효율, 자세 안정성 등
- 지형: 평지 또는 다양한 지형(계단, 경사 등)

**WholeBodyTrackingManager** — 모션 클립 추적
- 명령: `.npz` 모션 파일에서 프레임 단위로 제공
- 보상: 관절 위치/방향 오차, 속도 오차 등
- 모션 클립의 각 프레임을 따라가도록 학습

---

### 4.2 `holosoma_inference` — 추론/배포

학습이 끝난 정책(ONNX 모델)을 **실제 로봇** 또는 **MuJoCo 시뮬레이션**에서 실행하는 패키지.

```
학습 완료 → ONNX 내보내기 → holosoma_inference로 실행
```

**주요 구성요소:**
- `run_policy.py` — 추론 실행 진입점
- `policies/locomotion.py` — 보행 추론 정책
- `policies/wbt.py` — WBT 추론 정책
- `models/` — 사전학습된 ONNX 모델 포함
- `sdk/` — 실제 로봇과 통신하는 SDK 브릿지

**배포 워크플로우 문서:**
- `docs/workflows/real-robot-locomotion.md` — 실제 로봇 보행
- `docs/workflows/real-robot-wbt.md` — 실제 로봇 WBT
- `docs/workflows/sim-to-sim-locomotion.md` — MuJoCo에서 보행 테스트
- `docs/workflows/sim-to-sim-wbt.md` — MuJoCo에서 WBT 테스트

---

### 4.3 `holosoma_retargeting` — 모션 리타겟팅

사람의 모션 캡처 데이터를 로봇이 실행할 수 있는 형태로 변환하는 패키지.

**왜 필요한가?**
- 사람과 로봇은 체형이 다르다 (팔 길이, 관절 개수 등)
- 사람 모션을 그대로 로봇에 넣으면 물리적으로 불가능한 자세가 됨
- 리타겟팅이 이를 로봇의 물리적 제약에 맞게 변환

**파이프라인:**
```
① 원본 모캡 데이터 (AMASS/LAFAN/OMOMO)
        ↓
② 전처리 (data_utils/prep_amass_smplx_for_rt.py)
        ↓
③ 리타겟팅 (examples/robot_retarget.py)
   - InteractionMeshRetargeter: 메쉬 기반 최적화로 변환
        ↓
④ 데이터 변환 (data_conversion/convert_data_format_mj.py)
   - MuJoCo 역운동학으로 관절 각도 계산
   - .npz 형식으로 저장 (학습에 사용할 포맷)
        ↓
⑤ 학습에 사용 (holosoma의 WBT command로 전달)
```

**지원 데이터 포맷:**
- AMASS SMPL-X / SMPL-H
- LAFAN
- OMOMO (사람-물체 상호작용)

---

## 5. Manager 기반 아키텍처

Holosoma의 핵심 설계 패턴. 환경의 각 측면을 독립적인 **Manager**로 분리했다.

```
┌─────────────────────────────────────────────────────────────┐
│                        BaseTask                             │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Observation   │  │   Action     │  │    Reward        │  │
│  │ Manager       │  │   Manager    │  │    Manager       │  │
│  │               │  │              │  │                  │  │
│  │ - 관절 각도   │  │ - 정책 출력  │  │ - 위치 추적 보상 │  │
│  │ - 관절 속도   │  │   → PD 제어  │  │ - 에너지 페널티  │  │
│  │ - IMU (자세)  │  │   → 토크 명령│  │ - 안정성 보상    │  │
│  │ - 명령 정보   │  │              │  │ - ...            │  │
│  │ - 과거 히스토리│  │              │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Command      │  │ Termination  │  │ Randomization    │  │
│  │  Manager      │  │ Manager      │  │ Manager          │  │
│  │               │  │              │  │                  │  │
│  │ Loco: 속도명령│  │ - 넘어짐     │  │ - 마찰 계수 변경 │  │
│  │ WBT: 모션클립 │  │ - 시간 초과  │  │ - 질량 변경      │  │
│  │               │  │ - 관절 한계  │  │ - 외부 힘 (밀기) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Curriculum   │  │   Terrain    │  │  Reset Event     │  │
│  │  Manager      │  │   Manager    │  │  Manager         │  │
│  │               │  │              │  │                  │  │
│  │ - 난이도 조절 │  │ - 평지       │  │ - 초기 자세 설정 │  │
│  │ - 보상 스케줄 │  │ - 계단/경사  │  │ - 관절 초기화    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 각 Manager의 역할

| Manager | 역할 | 설정 위치 |
|---------|------|----------|
| **ObservationManager** | 정책에 전달할 관측값 구성. 히스토리 버퍼링 지원 | `config_values/*/observation.py` |
| **ActionManager** | 정책 출력 → 관절 명령 변환. PD 제어 적용 | `config_values/*/action.py` |
| **RewardManager** | 보상 항목(term)들의 가중합 계산 | `config_values/*/reward.py` |
| **CommandManager** | Loco: 속도 목표 생성 / WBT: 모션 클립 프레임 공급 | `config_values/*/command.py` |
| **TerminationManager** | 에피소드 종료 조건 판단 (넘어짐, 시간 초과 등) | `config_values/*/termination.py` |
| **RandomizationManager** | 환경 파라미터 무작위화 (sim-to-real 핵심) | `config_values/*/randomization.py` |
| **CurriculumManager** | 학습 진행에 따라 난이도 조절 | `config_values/*/curriculum.py` |
| **TerrainManager** | 지형 생성 및 로봇 배치 | `config_values/terrain.py` |
| **ResetEventManager** | 환경 리셋 시 초기 상태 설정 | 시뮬레이터 설정 내 |

### Manager Term 구조

각 Manager는 여러 **Term**(항목)으로 구성된다. 예를 들어 RewardManager:

```python
# config_values/wbt/g1/reward.py (간소화 예시)
reward_terms = {
    "motion_global_ref_position_error_exp": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:...",
        weight=5.0,    # 가중치
    ),
    "motion_joint_position_error": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:...",
        weight=2.0,
    ),
    "energy_penalty": RewardTermCfg(
        func="holosoma.managers.reward.terms.common:...",
        weight=-0.001,  # 음수 = 페널티
    ),
}
```

각 term은 `func` 필드로 Python 함수/클래스를 지정한다. 이렇게 하면 새로운 보상 항목을 추가하거나 교체하기가 매우 쉽다.

---

## 6. 학습 루프 상세 흐름

하나의 스텝(`env.step()`)이 실행될 때 내부에서 벌어지는 일:

```
env.step(actions)
│
├── 1. _pre_physics_step(actions)
│   └── ActionManager.process_actions()    # 정책 출력 → 관절 목표 변환
│
├── 2. _physics_step()
│   └── for _ in range(control_decimation):  # 물리 시뮬 여러 번 반복
│       ├── ActionManager.apply_actions()    # PD 제어로 토크 계산 & 적용
│       └── simulator.simulate()             # 물리 시뮬레이션 1스텝
│
└── 3. _post_physics_step()
    ├── simulator.refresh_sim_tensors()      # 시뮬레이터 상태 읽기
    ├── episode_length_buf += 1
    │
    ├── CommandManager.step()                # 명령 업데이트
    ├── CurriculumManager.step()             # 난이도 업데이트
    ├── RandomizationManager.step()          # 랜덤화 적용
    │
    ├── TerminationManager.check()           # 종료 조건 확인
    ├── RewardManager.compute()              # 보상 계산
    │
    ├── [에피소드 종료된 환경 리셋]
    │   ├── ObservationManager.reset()
    │   ├── 모든 Manager.reset()
    │   └── ResetEventManager.reset_scene()
    │
    └── ObservationManager.compute()         # 다음 관측값 계산
```

**control_decimation**: 정책이 1번 행동을 결정하면, 물리 시뮬레이션은 여러 번(보통 4~10번) 돌린다. 이는 실제 로봇의 제어 주파수와 물리 시뮬레이션 주파수가 다르기 때문이다.

---

## 7. 설정(Config) 시스템

Holosoma는 **Tyro** 라이브러리 기반의 계층적 설정 시스템을 사용한다.

### 구조

```
config_types/    ← 설정의 "스키마" (어떤 필드가 있는지 정의, dataclass)
config_values/   ← 설정의 "값" (실제 프리셋 조합)
```

### ExperimentConfig (최상위 설정)

```python
@dataclass
class ExperimentConfig:
    training: TrainingConfig          # 학습 파라미터 (num_envs, seed 등)
    env_class: str                    # 환경 클래스 경로
    algo: AlgoConfig                  # 알고리즘 설정 (PPO or FastSAC)
    simulator: SimulatorConfig        # 시뮬레이터 설정
    robot: RobotConfig                # 로봇 정보 (URDF, 관절 이름, PD 게인 등)
    terrain: TerrainConfig            # 지형 설정
    observation: ObservationConfig    # 관측 설정
    action: ActionConfig              # 행동 설정
    reward: RewardConfig              # 보상 설정
    termination: TerminationConfig    # 종료 조건 설정
    randomization: RandomizationConfig # 랜덤화 설정
    command: CommandConfig            # 명령 설정
    curriculum: CurriculumConfig      # 커리큘럼 설정
    logger: LoggerConfig              # 로깅 설정 (wandb, tensorboard 등)
```

### CLI 사용법

```bash
python train_agent.py \
    exp:g1-29dof-wbt \          # 프리셋 선택 (config_values의 조합)
    simulator:isaacgym \         # 시뮬레이터 선택
    logger:wandb \               # 로깅 방식
    --training.num_envs 4096 \   # 개별 파라미터 오버라이드
    --training.seed 42 \
    --algo.config.num_learning_iterations 50000
```

`exp:g1-29dof-wbt` 같은 프리셋은 `config_values/wbt/g1/experiment.py`에 정의되어 있고, 알고리즘·로봇·보상 등 모든 설정을 한 번에 지정한다. CLI에서 `--` 옵션으로 개별 값을 덮어쓸 수 있다.

---

## 8. 지원 로봇/시뮬레이터/알고리즘

### 로봇

| 로봇 | DOF | 제조사 | 설명 |
|------|-----|--------|------|
| **Unitree G1** | 29 | Unitree | 29자유도 휴머노이드 |
| **Booster T1** | 29 | Booster Robotics | 29자유도 휴머노이드 |

로봇 URDF 파일: `src/holosoma/holosoma/data/robots/g1/`, `t1/`

### 시뮬레이터

| 시뮬레이터 | 용도 | GPU 가속 | 비고 |
|-----------|------|---------|------|
| **IsaacGym** | 학습 | O | NVIDIA, 수천 환경 병렬 |
| **IsaacSim** | 학습 | O | Omniverse 기반, 렌더링 우수 |
| **MJWarp** | 학습 | O | GPU 가속 MuJoCo (베타) |
| **MuJoCo** | 추론만 | X | 크로스 시뮬레이터 검증용 |

### RL 알고리즘

| 알고리즘 | 타입 | 특징 |
|---------|------|------|
| **PPO** | On-policy | 안정적, 범용적. 기본 40,000 iteration |
| **FastSAC** | Off-policy | 샘플 효율 높음. 기본 400,000 iteration |

---

## 9. 데이터 파이프라인: 모션 데이터 → 학습

WBT 학습을 위해 모션 데이터가 어떻게 흘러가는지 단계별로:

### Step 1: 모션 캡처 데이터 준비

원본 데이터 형식: AMASS(SMPL-X), LAFAN, OMOMO 등

### Step 2: 리타겟팅

```bash
cd src/holosoma_retargeting/
python examples/robot_retarget.py \
    --data_path <모캡_데이터_경로> \
    --task-type robot_only \
    --task-name <시퀀스_이름> \
    --data_format smplh
```

결과: 로봇 관절 공간으로 변환된 모션 파일

### Step 3: 데이터 변환 (학습용 포맷)

```bash
python data_conversion/convert_data_format_mj.py \
    --input_file <리타겟팅_결과.npz> \
    --output_fps 50 \
    --output_name <변환결과.npz> \
    --data_format smplh
```

결과: `.npz` 파일 — 학습에 바로 사용 가능한 포맷

### Step 4: 학습

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    simulator:isaacsim \
    --command.setup_terms.motion_command.params.motion_config.motion_file=<변환결과.npz>
```

### 모션 .npz 파일의 내부 구조

학습에 사용되는 `.npz` 파일에는 대략 다음 정보가 들어있다:

| 키 | 설명 |
|----|------|
| `root_pos` | 루트(골반) 위치 `(T, 3)` |
| `root_quat` | 루트 회전 `(T, 4)` |
| `joint_pos` | 관절 각도 `(T, num_dof)` |
| `joint_vel` | 관절 속도 `(T, num_dof)` |
| `body_pos` | 각 바디 파트의 3D 위치 `(T, num_bodies, 3)` |
| `body_quat` | 각 바디 파트의 회전 `(T, num_bodies, 4)` |
| `fps` | 프레임 레이트 |

`T` = 총 프레임 수, `num_dof` = 관절 자유도 수 (G1은 29)

---

## 10. 실행 방법 요약

### 환경 설정

```bash
# IsaacGym 학습 환경
bash scripts/setup_isaacgym.sh

# IsaacSim 학습 환경
bash scripts/setup_isaacsim.sh

# MuJoCo/MJWarp 환경
bash scripts/setup_mujoco.sh

# 추론 전용
bash scripts/setup_inference.sh

# 리타겟팅
bash scripts/setup_retargeting.sh
```

### Locomotion 학습

```bash
source scripts/source_isaacgym_setup.sh

# G1 로봇, PPO
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof simulator:isaacgym logger:wandb --training.seed 1

# G1 로봇, FastSAC
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac simulator:isaacgym logger:wandb
```

### WBT 학습

```bash
source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=<모션파일.npz>
```

### 전체 파이프라인 데모 (리타겟팅 → 학습 한번에)

```bash
# OMOMO 데이터로 전체 파이프라인
bash demo_scripts/demo_omomo_wb_tracking.sh

# LAFAN 데이터로 전체 파이프라인
bash demo_scripts/demo_lafan_wb_tracking.sh
```

### 추론 (학습 완료 후)

```bash
# MuJoCo에서 시뮬레이션 테스트
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-loco --task.model-path <ONNX_모델_경로>
```

---

## 11. 내 데이터를 넣으려면?

학부 인턴으로서 자신의 motion/scene 데이터를 넣어 WBT 정책을 학습하려면 아래 단계를 따르면 된다.

### 경우 1: 이미 로봇 관절 형식의 모션 데이터가 있다면

1. `.npz` 형식으로 변환 (위 Section 9의 모션 파일 구조 참고)
2. 필요한 키: `root_pos`, `root_quat`, `joint_pos`, `joint_vel`, `body_pos`, `body_quat`, `fps` 등
3. 관절 이름과 순서가 G1 로봇의 URDF와 일치해야 함
4. 학습 시 `--command.setup_terms.motion_command.params.motion_config.motion_file=<경로>` 로 지정

### 경우 2: 사람 모션 캡처 데이터가 있다면

1. SMPL-X/SMPL-H 형식으로 변환 (또는 이미 해당 형식이면 그대로)
2. `holosoma_retargeting`으로 리타겟팅 실행
3. `convert_data_format_mj.py`로 학습용 변환
4. 학습 실행

### 경우 3: Scene 데이터(물체 포함)가 있다면

1. 물체의 URDF 파일 준비 (3D 메쉬 + 물리 속성)
2. `exp:g1-29dof-wbt-w-object` 프리셋 사용
3. `robot.object.object_urdf_path` 에 물체 URDF 경로 지정
4. 리타겟팅 시 `--task-type interaction` 옵션 사용 (물체 상호작용 포함)

### 주의사항

- 모션 데이터의 FPS가 시뮬레이터 제어 주파수와 맞는지 확인 (보통 50 FPS)
- 관절 순서가 로봇 URDF의 관절 순서와 일치해야 함
- `config_values/wbt/g1/command.py`의 `body_names_to_track`이 추적할 바디 파트 목록 — 자신의 데이터에 맞게 조정 필요할 수 있음
- IsaacGym 또는 IsaacSim 설치가 필요 (NVIDIA GPU 필수)
