import utils
import eda
import modeling

def main():
    # ---------------------------------------------------------
    # 0. 기본 설정 및 데이터 경로 지정
    # ---------------------------------------------------------
    utils.set_environment()
    
    # 분석할 데이터 경로 리스트 (사용자 환경에 맞게 수정)
    file_paths = [
        '카드소비 데이터_수원시/소매_유통_2022_수원시.csv',
        '카드소비 데이터_수원시/소매_유통_2023_수원시.csv',
        '카드소비 데이터_수원시/소매_유통_2024_수원시.csv'
    ]
    target_category = '패션잡화' # 분석할 카테고리

    # ---------------------------------------------------------
    # 1. 데이터 전처리 파이프라인 (utils)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" 🛠️ [Phase 1] 데이터 준비 및 전처리 시작")
    print("="*50)
    
    df_raw = utils.load_and_filter_data(file_paths, target_category)
    daily = utils.aggregate_and_log_transform(df_raw)
    daily = utils.add_exog_variables(daily)
    daily = utils.detect_outliers(daily, window=7)

    # ---------------------------------------------------------
    # 2. 탐색적 데이터 분석 및 정상성 검정 (eda)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" 📊 [Phase 2] 데이터 탐색 및 정상성 검정")
    print("="*50)
    
    # 시계열 기본 형태 및 성분 분해 확인 (시각화 창이 뜹니다)
    eda.plot_time_series(daily, col='amt_log', title=f'[{target_category}] 일별 매출 시계열')
    eda.plot_stl_decomposition(daily, col='amt_log', period=7)
    eda.plot_rolling_stats(daily, col='amt_log', window=30)  # 롤링 평균/분산
    eda.plot_calendar_effects(daily, col='amt_log')    # 달력 효과 박스플롯
    
    # 원본 데이터 정상성 검정
    is_stationary = eda.check_stationarity(daily['amt_log'], diff_order=0)
    
    # 비정상적일 경우 1차 차분하여 확인 (여기서는 시각화만 수행)
    if not is_stationary:
        print("\n⚠️ 원본 시계열이 비정상적입니다. ACF/PACF에서 차분이 필요한지 확인하세요.")
        diff_data = daily['amt_log'].diff().dropna()
        eda.check_stationarity(diff_data, diff_order=1)
        eda.plot_acf_pacf_charts(diff_data, title_prefix="1차 차분")
    else:
        eda.plot_acf_pacf_charts(daily['amt_log'], title_prefix="원본")

    # ---------------------------------------------------------
    # 3. 모델링 및 블프 효과 검증 (modeling)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" ⚙️ [Phase 3] SARIMAX 모형 적합 및 검증")
    print("="*50)
    
    # 목표 변수(y)와 외생 변수(X) 설정
    y = daily['amt_log']
    X = daily[['is_bf', 'is_holiday']]

    # 최적의 SARIMAX 모형 찾기
    best_model = modeling.run_auto_arima(y, X, m=7)
    
    if best_model is not None:
        # 모델 진단 (잔차 백색잡음/정규성 확인)
        modeling.diagnose_model(best_model)
        
        # ⭐ 블랙프라이데이 효과 통계적 검증
        modeling.verify_event_effect(best_model, event_col='is_bf', event_name='블랙프라이데이')
        modeling.verify_event_effect(best_model, event_col='is_holiday', event_name='공휴일')

        # 모델 과적합 방지용 Test (20% 데이터로 미래 예측 성능 평가)
        modeling.validate_overfitting(y, X, m=7, test_ratio=0.2)
        
if __name__ == "__main__":
    main()