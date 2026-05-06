"""
환경 설정, 데이터 로드 및 전처리, 파생변수(외생변수) 생성, 이상치 탐지
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import holidays

def set_environment():
    """Step 0: 경고 무시 및 시각화 한글 폰트 설정"""
    warnings.filterwarnings('ignore')
    
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
    except:
        try:
            plt.rcParams['font.family'] = 'AppleGothic' # Mac
        except:
            print("경고: 글꼴 설정에 실패했습니다. 그래프에서 한글이 깨질 수 있습니다.")
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 환경 설정 완료 (한글 폰트 적용)")

def load_and_filter_data(file_paths, category_name):
    """Step 1: 데이터를 로드하고 특정 카테고리로 필터링"""
    df_list = [pd.read_csv(path, encoding='cp949') for path in file_paths]
    df = pd.concat(df_list, ignore_index=True)
    df['ta_ymd'] = pd.to_datetime(df['ta_ymd'])
    
    # 카테고리 필터링
    df_filtered = df.loc[df['card_tpbuz_nm_2'] == category_name].copy()
    
    print(f"✅ 데이터 로드 및 필터링 완료: {category_name} (형태: {df_filtered.shape})")
    return df_filtered

def aggregate_and_log_transform(df):
    """Step 1-1 & 1-2: 일별 집계 및 로그 변환"""
    daily = df.groupby(['ta_ymd', 'day']).agg({
        'amt': 'sum',
        'cnt': 'sum'
    }).reset_index('day')
    
    daily = daily.fillna(0) # 결측치 처리
    
    # 로그 변환
    daily['amt_log'] = np.log1p(daily['amt'])
    daily['cnt_log'] = np.log1p(daily['cnt'])
    
    # AOV (거래당 평균금액)
    daily['aov'] = daily['amt'] / (daily['cnt'] + 1e-9)
    daily['aov_log'] = np.log1p(daily['aov'])
    
    print(f"✅ 일별 집계 및 로그 변환 완료 (최종 행 수: {len(daily)})")
    return daily

def add_exog_variables(daily):
    """Step 2: 블랙프라이데이 및 공휴일 외생변수(더미) 추가"""
    # 1. 블프 기간 정의 (하드코딩된 기간을 리스트로 관리)
    bf_periods = [
        pd.date_range('2022-11-14', '2022-11-27', freq='D'),
        pd.date_range('2023-11-13', '2023-11-26', freq='D'),
        pd.date_range('2024-11-18', '2024-12-01', freq='D')
    ]
    
    daily['is_bf'] = 0
    for bf in bf_periods:
        mask = (daily.index >= bf[0]) & (daily.index <= bf[-1])
        daily.loc[mask, 'is_bf'] = 1
        
    # 2. 공휴일 더미 생성
    kr_holidays = holidays.KR(years=range(2022, 2025))
    daily['is_holiday'] = daily.index.map(lambda x: x in kr_holidays).astype(int)
        
    print(f"✅ 외생변수 추가 완료 (블프: {daily['is_bf'].sum()}일, 공휴일: {daily['is_holiday'].sum()}일)")
    return daily

def detect_outliers(daily, window=7):
    """Step 3: 롤링 MAD 기반 이상치 탐지"""
    s = pd.Series(daily['amt_log'].values, index=daily.index).astype(float)
    
    mu = s.rolling(window, center=True, min_periods=3).median()
    dev = (s - mu).abs()
    mad = dev.rolling(window, center=True, min_periods=3).median()
    
    rob_z = (s - mu) / (1.4826 * mad.replace(0, np.nan))
    rob_z = rob_z.fillna(0)
    
    daily['rob_z'] = rob_z
    daily['is_outlier'] = (rob_z.abs() > 4).astype(int)
    
    outlier_count = daily['is_outlier'].sum()
    print(f"✅ 이상치 탐지 완료 (탐지된 이상치: {outlier_count}개)")
    
    return daily