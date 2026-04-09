"""전체 파이프라인 실행: 데이터 준비 → 특성 추출 → 모델 학습 → 저장"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import load_or_generate_data, cycles_to_dataframe
from src.features import extract_all_features, get_feature_columns
from src.model import train_and_compare, save_model, get_feature_importance


def main():
    os.makedirs('outputs', exist_ok=True)

    # ========== Phase 1: 데이터 준비 ==========
    print("\n" + "=" * 60)
    print("Phase 1: 데이터 준비")
    print("=" * 60)

    all_cells = load_or_generate_data('data')
    summary_df = cycles_to_dataframe(all_cells)
    print(f"\n총 셀: {len(all_cells)}개")
    print(f"총 사이클: {len(summary_df)}개")
    print(f"\nSOH 범위: {summary_df['soh'].min():.1f}% ~ {summary_df['soh'].max():.1f}%")

    # 용량 감소 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cell in summary_df['cell'].unique():
        cell_df = summary_df[summary_df['cell'] == cell]
        axes[0].plot(cell_df['cycle'], cell_df['capacity'], label=cell, alpha=0.8)
    axes[0].set_xlabel('Cycle Number')
    axes[0].set_ylabel('Capacity (Ah)')
    axes[0].set_title('Capacity Degradation over Cycles')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for cell in summary_df['cell'].unique():
        cell_df = summary_df[summary_df['cell'] == cell]
        axes[1].plot(cell_df['cycle'], cell_df['soh'], label=cell, alpha=0.8)
    axes[1].set_xlabel('Cycle Number')
    axes[1].set_ylabel('SOH (%)')
    axes[1].set_title('SOH Degradation over Cycles')
    axes[1].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='EOL Threshold (80%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/capacity_degradation.png', dpi=150)
    plt.close()
    print("  → outputs/capacity_degradation.png 저장")

    # 충방전 곡선 시각화 (첫 번째 셀, 초기/중기/후기)
    first_cell = list(all_cells.keys())[0]
    charge_cycles = all_cells[first_cell]['charge']
    n = len(charge_cycles)
    sample_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx in sample_indices:
        cyc = charge_cycles[idx]
        label = f"Cycle {cyc['cycle_number']}"
        axes[0].plot(cyc['Time'], cyc['Voltage_measured'], label=label, alpha=0.8)
    axes[0].set_xlabel('Time (min)')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_title(f'{first_cell} - Charge Voltage Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    discharge_cycles = all_cells[first_cell]['discharge']
    for idx in sample_indices:
        cyc = discharge_cycles[min(idx, len(discharge_cycles) - 1)]
        label = f"Cycle {cyc['cycle_number']}"
        axes[1].plot(cyc['Time'], cyc['Voltage_measured'], label=label, alpha=0.8)
    axes[1].set_xlabel('Time (min)')
    axes[1].set_ylabel('Voltage (V)')
    axes[1].set_title(f'{first_cell} - Discharge Voltage Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/voltage_curves.png', dpi=150)
    plt.close()
    print("  → outputs/voltage_curves.png 저장")

    # ========== Phase 2: 특성 추출 ==========
    print("\n" + "=" * 60)
    print("Phase 2: 특성 추출")
    print("=" * 60)

    # 충전과 방전 모두에서 특성 추출
    charge_features = extract_all_features(all_cells, 'charge')
    discharge_features = extract_all_features(all_cells, 'discharge')

    print(f"충전 특성: {charge_features.shape}")
    print(f"방전 특성: {discharge_features.shape}")

    # 충전 데이터 기반으로 모델 학습 (짧은 충전 측정으로 SOH 예측이 목표)
    df = charge_features.copy()
    feature_cols = get_feature_columns(df)
    print(f"\n추출된 특성 ({len(feature_cols)}개):")
    for col in feature_cols:
        print(f"  - {col}")

    # 특성 테이블 미리보기
    print(f"\n특성 테이블 미리보기:")
    preview_cols = ['cell', 'cycle', 'voltage_slope', 'time_3_5_to_4_0',
                    'temp_max', 'internal_resistance', 'soh']
    available = [c for c in preview_cols if c in df.columns]
    print(df[available].head(10).to_string(index=False))

    # 특성 저장
    df.to_csv('outputs/features.csv', index=False)
    print("\n  → outputs/features.csv 저장")

    # 특성 상관관계 히트맵
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[feature_cols + ['soh']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, square=True, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=150)
    plt.close()
    print("  → outputs/correlation_heatmap.png 저장")

    # ========== Phase 3: 모델 학습 ==========
    print("\n" + "=" * 60)
    print("Phase 3: 모델 학습")
    print("=" * 60)

    model_data = train_and_compare(df, feature_cols)

    # 특성 중요도
    imp_df = get_feature_importance(model_data['best']['model'], feature_cols)
    print(f"\n특성 중요도 (Top 10):")
    print(imp_df.head(10).to_string(index=False))

    # 예측 vs 실제 비교 그래프
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (name, key) in zip(axes, [('Random Forest', 'rf'), ('XGBoost', 'xgb')]):
        res = model_data[key]['results']
        ax.scatter(res['y_test'], res['y_pred'], alpha=0.5, s=20)
        ax.plot([50, 105], [50, 105], 'r--', alpha=0.7)
        ax.set_xlabel('Actual SOH (%)')
        ax.set_ylabel('Predicted SOH (%)')
        ax.set_title(f'{name}\nMAE={res["MAE"]:.2f}%, RMSE={res["RMSE"]:.2f}%, R²={res["R2"]:.4f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/prediction_comparison.png', dpi=150)
    plt.close()
    print("  → outputs/prediction_comparison.png 저장")

    # 특성 중요도 그래프
    fig, ax = plt.subplots(figsize=(10, 6))
    top_imp = imp_df.head(10)
    ax.barh(range(len(top_imp)), top_imp['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp['feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 10 Feature Importance ({model_data["best"]["name"]})')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=150)
    plt.close()
    print("  → outputs/feature_importance.png 저장")

    # 모델 저장
    save_model(model_data)

    # 샘플 예측 데이터 생성 (대시보드 테스트용)
    sample_cell = list(all_cells.keys())[0]
    sample_cycle = all_cells[sample_cell]['charge'][50]
    import pandas as pd
    sample_df = pd.DataFrame({
        'Time': sample_cycle['Time'],
        'Voltage_measured': sample_cycle['Voltage_measured'],
        'Current_measured': sample_cycle['Current_measured'],
        'Temperature_measured': sample_cycle['Temperature_measured'],
    })
    sample_df.to_csv('data/sample_input.csv', index=False)
    print("\n  → data/sample_input.csv 저장 (대시보드 테스트용)")

    print("\n" + "=" * 60)
    print("파이프라인 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
