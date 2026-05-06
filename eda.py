"""
EDA 및 시계열 정상성 검정
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_time_series(daily, col='amt_log', title='일별 매출 시계열'):
    """Step 4-1: 기본적인 시계열 플롯 시각화"""
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(daily.index, daily[col], color='#11224E', alpha=0.8, linewidth=0.8)
    ax.set_xlabel('날짜', fontsize=14)
    ax.set_ylabel(col, fontsize=14)
    ax.set_title(title, fontsize=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_stl_decomposition(daily, col='amt_log', period=7):
    """Step 4-2: STL 분해 (추세 + 계절 + 잔차) 시각화 및 범위 출력"""
    stl = STL(daily[col], period=period, robust=True).fit()
    
    fig = stl.plot()
    fig.set_size_inches(15, 10)
    axes = fig.axes
    
    axes[0].set_title('원본 시계열', fontsize=16)
    axes[1].set_title('추세 성분', fontsize=16)
    axes[2].set_title('계절 성분 (주간)', fontsize=16)
    axes[3].set_title('잔차 성분', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 분해 성분 범위 출력
    trend, seasonal, resid = stl.trend, stl.seasonal, stl.resid
    print(f"\n【분해 성분 통계】")
    print(f"  추세 범위: {trend.min():.3f} ~ {trend.max():.3f} (격차: {trend.max()-trend.min():.3f})")
    print(f"  계절 범위: {seasonal.min():.3f} ~ {seasonal.max():.3f} (격차: {seasonal.max()-seasonal.min():.3f})")
    print(f"  잔차 범위: {resid.min():.3f} ~ {resid.max():.3f} (격차: {resid.max()-resid.min():.3f})")

def plot_rolling_stats(daily, col='amt_log', window=30):
    """Step 4-3: 롤링 평균/분산을 통한 정상성 시각화"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # 롤링 평균
    rolling_mean = daily[col].rolling(window).mean()
    axes[0].plot(daily.index, daily[col], label='원본', color='#11224E', alpha=0.7, linewidth=0.8)
    axes[0].plot(daily.index, rolling_mean, label=f'{window}일 롤링 평균', linewidth=1.5, color='red')
    axes[0].set_title('롤링 평균 (평균 일정성 확인)', fontsize=18)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 롤링 표준편차
    rolling_std = daily[col].rolling(window).std()
    axes[1].plot(daily.index, rolling_std, label=f'{window}일 롤링 표준편차', color='orange', linewidth=2)
    axes[1].axhline(y=rolling_std.mean(), color='red', linestyle='--', label='평균')
    axes[1].set_title('롤링 분산 (분산 일정성 확인)', fontsize=18)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# eda.py 파일의 맨 아래에 추가해 주세요.

def plot_calendar_effects(daily, col='amt_log'):
    """Step 4-4: 달력 효과 (요일별 및 이벤트별 매출 분포 박스플롯)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) 요일별 분포
    # 주의: daily 데이터프레임에 'day' 컬럼(1~7)이 존재해야 합니다.
    dow_data = [daily[daily['day']==i][col].values for i in range(1, 8)]
    axes[0].boxplot(dow_data, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[0].set_ylabel(f'{col}', fontsize=14)
    axes[0].set_title('요일별 매출 분포', fontsize=18)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', labelsize=12)

    # 2) 블프 vs 공휴일 vs 평시 분포
    bf_data = daily[(daily['is_bf'] == 0) & (daily['is_holiday'] == 0)][col].values
    nbf_data = daily[(daily['is_bf'] == 1) & (daily['is_holiday'] == 0)][col].values
    hol_data = daily[(daily['is_bf'] == 0) & (daily['is_holiday'] == 1)][col].values
    
    axes[1].boxplot([bf_data, nbf_data, hol_data], labels=['평시', '블프', '공휴일'])
    axes[1].set_ylabel(f'{col}', fontsize=14)
    axes[1].set_title('블프 vs 공휴일 vs 평시', fontsize=18)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', labelsize=12)

    plt.tight_layout()
    plt.show()

def check_stationarity(series, diff_order=0):
    """Step 5: ADF 및 KPSS 검정을 통한 시계열 정상성 확인"""
    # 결측치 제거 후 검정
    clean_series = series.dropna()
    
    adf_stat, adf_p = adfuller(clean_series)[:2]
    kpss_stat, kpss_p = kpss(clean_series)[:2]
    
    title = "원본" if diff_order == 0 else f"{diff_order}차 차분 후"
    
    print(f"\n【{title} 정상성 검정】")
    print(f"  ADF: stat = {adf_stat:.4f}, p-value = {adf_p:.4f} → {'정상 (p<=0.05)' if adf_p <= 0.05 else '비정상'}")
    print(f"  KPSS: stat = {kpss_stat:.4f}, p-value = {kpss_p:.4f} → {'정상 (p>0.05)' if kpss_p > 0.05 else '비정상'}")
    
    return adf_p <= 0.05 and kpss_p > 0.05

def plot_acf_pacf_charts(series, lags=40, title_prefix="원본"):
    """Step 6: ACF 및 PACF 시각화를 통한 ARIMA 차수(p, q) 식별"""
    clean_series = series.dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(clean_series, lags=lags, ax=axes[0])
    axes[0].set_title(f'ACF ({title_prefix})', fontsize=18)
    
    plot_pacf(clean_series, lags=lags, ax=axes[1])
    axes[1].set_title(f'PACF ({title_prefix})', fontsize=18)
    
    plt.tight_layout()
    plt.show()