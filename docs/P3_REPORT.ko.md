# P3 로컬 공개 후보 검증

[English](P3_REPORT.md) · [공통 상태 JSON](../publication/P3_STATUS.json) · [새 재현 receipt](../publication/clean_reproduction.json)

상태: scientific_scope=P1_SPECIAL_CASE_AND_P2_SINGLE_GAUSSIAN_SETTING, clean_public_copy=PASS, public_reproduction=PASS, original_preservation=PASS, publication_readiness=LOCAL_PUBLIC_CANDIDATE_READY, remote_publication=NOT_RUN입니다. 이번 로컬 단계의 BLOCKED 항목은 없습니다. 원격 CI·게시는 의도적으로 미실행입니다.

새 공개 파일 복사본과 Python 3.11.15의 새 venv를 P1/P2와 같은 Linux/WSL2 머신·OS에서 실행했습니다. 시스템 Python3.11이 없어 기존 Python3.11 runtime을 읽기 전용으로 이용해 venv를 만들었고, 프로젝트·NumPy·Matplotlib는 모두 새 venv에서 import했습니다. 원 환경의 site-packages를 사용하지 않았습니다. clean copy에서 non-editable 설치했으며 `-I -B`, user site 비활성, include-system-site-packages=false, sys.path·.pth·import 경로·`pip check`를 확인했습니다([환경 receipt](../publication/clean_environment.json)). 다른 머신의 외부 독립 재현은 아닙니다.

선언한 CPU 직접·전이·build 의존성만 새 venv에 설치했습니다. NumPy=1.26.4,Matplotlib=3.10.9입니다. 기본 sandbox에서 PyPI DNS가 처음 실패했으나, 승인된 새 venv 전용 설치 접근으로 성공했습니다. 기존 환경 설치·업그레이드는 0입니다. 의존성 다운로드·설치 wall은 별도로 계측하지 않았으며 아래 저장 기록 검사 시간에 포함되지 않습니다.

clean copy root에서 새 venv Python으로 실제 실행한 명령은 `python -I -B -m teacher_error_geometry verify-public --output <NEW_EXTERNAL_OUTPUT>`입니다. P1 17개+P2 12개+공개 압축/도구 6개 테스트가 통과했습니다. 작은 fixture 검사 비용은 과학 full과 별도입니다. 기존 17개 P1 검사를 다시 호출했다고 고유 검사 수를 34개로 세지 않습니다.

257개 고유 일반 tar member에서 530파일·논리 크기 143,834,303bytes를 복원했습니다. archive hash·각 파일 bytes/SHA·두 최종 실행 식별자·P1 최초 동결 소스·동결 config·일반 공개 사본을 대조했습니다. P1 최초/최종 raw manifest가 같고 검사 후 복원 증거 전체 hash도 원형과 일치합니다. 복원 run 안에 새 receipt를 쓰지 않았습니다.

저장된 P1 168,000행과 P2 80,000행 전체의 완전성·고유 key·pairing·모집단 risk 항등식을 검사했습니다. 각 단계 사전 선택 dataset 12개를 재생성했습니다. P1 검산 lstsq는 32회, P2는 lstsq 60회(네 방식 재생성 48회+선형성 12회), QR 48회입니다. unit fixture 호출은 추가이며 합계는 별도로 계측하지 않았습니다. **신규 full sweep은 0회**입니다. P2의 과거 P1 보호 목록은 복원 경로로 대응시켜 원래 340개 bytes/hash를 모두 확인했고 과거 receipt는 수정하지 않았습니다.

동결 소스를 실행한 두 worker는 원 연구·인계 경로 open과 네트워크 socket 연결을 금지했습니다. 원본 경로 접근 시도는 0회였습니다. P2 과거 보호 목록의 경로 조회만 대응시켰고 수치 코드·기준은 바꾸지 않았습니다([이식성 변경](PORTABILITY.md)). 새 [P1](../publication/p1_public_verification.json)·[P2](../publication/p2_public_verification.json) receipt는 과거 receipt와 구분됩니다.

summary 9파일은 원 기록과 byte 동일하며 수치 최대 차이는 0입니다. 기준 atol=1e-10,rtol=1e-9는 유지했습니다. 공통 역사적 facts도 대조했습니다. 그림 7종×PNG/SVG=14파일 역시 **이번 환경에서** byte 동일했습니다. AI가 PNG 7개 모두의 축·범례·24개 각도 셀·양의 RM signed cross를 시각 점검했고 SVG 7개는 XML/label을 확인했습니다. 별도 SVG browser 렌더링·사람 심사는 수행하지 않았습니다. 환경 간 font/renderer bytes 동일성은 보장하지 않습니다.

통합 driver wall=32.477549896997516초, 자식 포함 CPU=32.3207629초입니다. 이를 포함한 `/usr/bin/time` 명령 전체는 wall 32.61초, CPU 32.45초, 최대 RSS 386.03125MiB입니다. 두 wall을 합산하지 않습니다. archive 생성은 별도로 21.639037539000128초였습니다. 의존성 설치·문서 작성·후속 파일/링크 검토는 이 시간에서 제외하며 하드웨어 정규화 benchmark가 아닙니다.

원본은 P1 full 두 번·P2·snapshot·보고서·이력·raw·config·과학 모듈·기존 인계·외부 P2 재생성을 포함한 558파일·144,133,880bytes를 시작 목록과 대조해 변경/추가/누락 0입니다. 원본 Git HEAD·branch·status·remote도 같습니다. 기존 P2의 340파일 보호 기록을 그대로 유지하면서 더 넓은 P3 목록으로 검사했습니다([보존 receipt](../publication/original_preservation.json)).

공개 진입 문서·wrapper에 개인 기본 경로는 없습니다. 역사적 receipt/facts의 원 경로 metadata는 hash 보존을 위해 그대로 둡니다. 새 P3 공개 receipt 사본은 경로 치환을 표시하고 원 local receipt hash를 함께 기록했습니다. [증거 manifest](../evidence/manifest.json)는 source/raw/receipt의 경로·bytes·hash·중복 대응·일반 파일 사본을 연결합니다. 공개 후보는 새 Git history이며 과거 실행 commit이 들어 있다고 주장하지 않고 frozen bytes로 확인합니다.

원 과학 범위·결과와 [P1 관측 후 진단 수정 이력](HISTORY.md)을 유지했습니다. n/lambda 선택·신규 MC 실험·GPU/model/유료 API·기존 환경 업그레이드·원격 게시 모두 0입니다. 다음 게시 지시에서 실제 원격 저장소 가용성·소유 상태를 확인합니다([release 제안](RELEASE_PLAN.md)). 현재 CI는 준비된 workflow이고 원격 CI PASS가 아닙니다.
