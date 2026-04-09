"""NASA PCoE 배터리 데이터 전처리 모듈"""
import numpy as np
import pandas as pd
import scipy.io
import os
import glob


def load_mat_file(filepath):
    """MATLAB .mat 파일에서 배터리 데이터 로드"""
    mat = scipy.io.loadmat(filepath, simplify_cells=True)
    # NASA 데이터셋 구조에서 배터리 데이터 추출
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    return mat[key]


def extract_cycles_from_mat(battery_data):
    """NASA .mat 파일에서 충방전 사이클 데이터 추출

    충전 사이클에는 Capacity가 없으므로, 바로 뒤따르는 방전 사이클의
    Capacity를 매핑하여 충전 사이클에도 SOH 레이블을 부여한다.
    """
    cycles = battery_data['cycle']
    charge_cycles = []
    discharge_cycles = []

    # 1차: 모든 사이클 파싱
    parsed = []
    for i, cycle in enumerate(cycles):
        cycle_type = cycle['type']
        if cycle_type not in ('charge', 'discharge'):
            continue
        data = cycle['data']

        cycle_dict = {
            'cycle_number': len(parsed) + 1,
            'type': cycle_type,
        }

        if isinstance(data, dict):
            for field in ['Voltage_measured', 'Current_measured',
                          'Temperature_measured', 'Time']:
                if field in data:
                    val = data[field]
                    if isinstance(val, np.ndarray):
                        cycle_dict[field] = val.astype(float)
                    else:
                        cycle_dict[field] = np.array([float(val)])

            if 'Capacity' in data:
                cap = data['Capacity']
                cycle_dict['Capacity'] = float(cap) if not isinstance(cap, np.ndarray) else float(cap.flat[0])

        parsed.append(cycle_dict)

    # 2차: 충전 사이클에 다음 방전 사이클의 Capacity 매핑
    for idx, cyc in enumerate(parsed):
        if cyc['type'] == 'charge' and 'Capacity' not in cyc:
            for j in range(idx + 1, min(idx + 3, len(parsed))):
                if parsed[j]['type'] == 'discharge' and 'Capacity' in parsed[j]:
                    cyc['Capacity'] = parsed[j]['Capacity']
                    break

    # 사이클 번호 재부여 (충방전 쌍 기준)
    charge_num = 0
    discharge_num = 0
    for cyc in parsed:
        if cyc['type'] == 'charge':
            charge_num += 1
            cyc['cycle_number'] = charge_num
            charge_cycles.append(cyc)
        elif cyc['type'] == 'discharge':
            discharge_num += 1
            cyc['cycle_number'] = discharge_num
            discharge_cycles.append(cyc)

    return charge_cycles, discharge_cycles


def generate_synthetic_battery_data(n_cells=4, max_cycles=200, seed=42):
    """NASA 데이터를 다운로드할 수 없을 때 사용할 합성 배터리 데이터 생성

    실제 리튬이온 배터리 열화 특성을 반영:
    - 용량은 사이클에 따라 비선형적으로 감소
    - 내부저항은 증가
    - 온도는 열화에 따라 약간 상승
    """
    np.random.seed(seed)
    all_cells = {}

    cell_names = [f'B{str(i).zfill(4)}' for i in range(5, 5 + n_cells)]
    initial_capacities = [2.0, 1.95, 2.05, 1.98][:n_cells]
    eol_cycles = [170, 150, 190, 160][:n_cells]

    for cell_idx, (cell_name, init_cap, eol) in enumerate(
            zip(cell_names, initial_capacities, eol_cycles)):
        charge_cycles = []
        discharge_cycles = []
        n_cycles = min(max_cycles, eol + 20)

        for cyc in range(1, n_cycles + 1):
            # 용량 감소 모델: 비선형 (제곱근 기반 + 노이즈)
            degradation = 1.0 - 0.3 * (cyc / eol) ** 1.5
            degradation = max(degradation, 0.5)
            capacity = init_cap * degradation + np.random.normal(0, 0.005)
            capacity = max(capacity, init_cap * 0.5)

            # 충전 사이클 데이터 생성
            dt = 0.1  # 시간 간격 (분)
            charge_time = np.arange(0, 60 + cyc * 0.05, dt)
            # 충전 전압 곡선 (CC-CV 패턴 근사)
            v_start = 3.0 + np.random.normal(0, 0.01)
            v_end = 4.2
            t_norm = charge_time / charge_time[-1]
            charge_voltage = v_start + (v_end - v_start) * (
                1 - np.exp(-4 * t_norm)) + np.random.normal(0, 0.002, len(charge_time))
            charge_voltage = np.clip(charge_voltage, 2.5, 4.2)

            charge_current = np.ones_like(charge_time) * 1.5
            # CV 단계에서 전류 감소
            cv_start = int(len(charge_time) * 0.7)
            charge_current[cv_start:] = 1.5 * np.exp(
                -3 * (t_norm[cv_start:] - t_norm[cv_start]))

            base_temp = 24.0 + cyc * 0.015
            charge_temp = base_temp + 5 * t_norm + np.random.normal(0, 0.3, len(charge_time))

            charge_cycles.append({
                'cycle_number': cyc,
                'type': 'charge',
                'Voltage_measured': charge_voltage,
                'Current_measured': charge_current,
                'Temperature_measured': charge_temp,
                'Time': charge_time,
                'Capacity': capacity,
            })

            # 방전 사이클 데이터 생성
            discharge_time = np.arange(0, 50 + cyc * 0.03, dt)
            t_norm_d = discharge_time / discharge_time[-1]
            # 방전 전압 곡선
            ir_drop = 0.02 + 0.0003 * cyc  # 내부저항 증가
            discharge_voltage = (4.2 - ir_drop) - (4.2 - 2.7) * (
                t_norm_d ** 0.8) + np.random.normal(0, 0.003, len(discharge_time))
            discharge_voltage = np.clip(discharge_voltage, 2.5, 4.2)

            discharge_current = -np.ones_like(discharge_time) * 2.0
            discharge_temp = base_temp + 8 * t_norm_d + np.random.normal(
                0, 0.3, len(discharge_time))

            discharge_cycles.append({
                'cycle_number': cyc,
                'type': 'discharge',
                'Voltage_measured': discharge_voltage,
                'Current_measured': discharge_current,
                'Temperature_measured': discharge_temp,
                'Time': discharge_time,
                'Capacity': capacity,
            })

        all_cells[cell_name] = {
            'charge': charge_cycles,
            'discharge': discharge_cycles,
            'initial_capacity': init_cap,
        }

    return all_cells


def load_or_generate_data(data_dir='data'):
    """데이터 로드: .mat 파일이 있으면 로드, 없으면 합성 데이터 생성"""
    mat_files = glob.glob(os.path.join(data_dir, '*.mat'))

    if mat_files:
        print(f"NASA .mat 파일 {len(mat_files)}개 발견. 로딩 중...")
        all_cells = {}
        for fpath in mat_files:
            cell_name = os.path.splitext(os.path.basename(fpath))[0]
            try:
                battery_data = load_mat_file(fpath)
                charge, discharge = extract_cycles_from_mat(battery_data)
                if discharge:
                    init_cap = discharge[0].get('Capacity', 2.0)
                    all_cells[cell_name] = {
                        'charge': charge,
                        'discharge': discharge,
                        'initial_capacity': init_cap,
                    }
                    print(f"  {cell_name}: {len(charge)} charge, {len(discharge)} discharge cycles")
            except Exception as e:
                print(f"  {cell_name} 로딩 실패: {e}")
        if all_cells:
            return all_cells

    print("NASA .mat 파일 없음 → 합성 배터리 데이터 생성 중...")
    return generate_synthetic_battery_data()


def cycles_to_dataframe(all_cells):
    """전체 셀 데이터를 요약 DataFrame으로 변환"""
    rows = []
    for cell_name, cell_data in all_cells.items():
        init_cap = cell_data['initial_capacity']
        for dc in cell_data['discharge']:
            cap = dc.get('Capacity', None)
            if cap is None:
                continue
            soh = (cap / init_cap) * 100.0
            rows.append({
                'cell': cell_name,
                'cycle': dc['cycle_number'],
                'capacity': cap,
                'soh': soh,
            })
    return pd.DataFrame(rows)
