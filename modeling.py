import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

def run_auto_arima(y, X, m=7, d=None, D=None):
    """Step 7-1: SARIMAX 최적 모형 탐색 및 적합"""
    print(f"\n--- Auto-ARIMA 탐색 시작 (데이터: {y.name}) ---")
    
    try:
        model_fit = pm.auto_arima(
            y, X=X, m=m,
            d=d, D=D, # 차분은 auto_arima가 찾도록
            start_p=0, start_q=0, max_p=2, max_q=2,
            seasonal=True, start_P=0, start_Q=0, max_P=2, max_Q=2,
            stepwise=True, trace=True, suppress_warnings=True
        )
        
        print("\n✨ 최적 모델 탐색 완료!")
        print(f"비계절 order (p,d,q): {model_fit.order}")
        print(f"계절 order (P,D,Q,m): {model_fit.seasonal_order}")
        
        return model_fit
    
    except Exception as e:
        print(f"❌ auto_arima 실행 중 오류 발생: {e}")
        return None

def diagnose_model(model_fit):
    """Step 7-2 ~ 7-5: 잔차 진단 및 통계적 검정 (Ljung-Box, Shapiro-Wilk)"""
    residuals = model_fit.resid()
    
    # 1. 잔차 시각화 진단
    try:
        fig = model_fit.plot_diagnostics(figsize=(15, 10), lags=30)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"그래프 생성 중 오류 발생: {e}")

    # 2. Ljung-Box 검정 (자기상관)
    lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    is_white_noise = all(lb_test['lb_pvalue'] > 0.05)
    
    print("\n【Ljung-Box 검정 (자기상관)】")
    print(f"  → {'잔차가 백색잡음 ✓ (모형 적절)' if is_white_noise else '자기상관 남음 ⚠️ (모형 개선 필요)'}")

    # 3. 정규성 검정
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print("\n【Shapiro-Wilk 정규성 검정】")
    print(f"  p-value: {shapiro_p:.4f} → {'정규분포 ✓' if shapiro_p > 0.05 else '정규분포 아님 (ARIMA는 robust하여 감안 가능)'}")

def verify_event_effect(model_fit, event_col='is_bf', event_name='블랙프라이데이'):
    """Step 8: 특정 외생변수(이벤트)의 매출 증감 효과 및 통계적 유의성 검증"""
    all_params = model_fit.params()
    all_pvalues = model_fit.pvalues()
    all_conf_int = model_fit.conf_int(alpha=0.05)
    
    if event_col not in all_params:
        print(f"❌ 모형에 '{event_col}' 변수가 없습니다.")
        return

    coef = all_params[event_col]
    pval = all_pvalues[event_col]
    ci = all_conf_int.loc[event_col]
    
    # 로그 스케일을 퍼센트로 변환
    pct_effect = (np.exp(coef) - 1) * 100
    ci_low = (np.exp(ci[0]) - 1) * 100
    ci_high = (np.exp(ci[1]) - 1) * 100

    print(f"\n【{event_name} 효과 검증】")
    print(f"  p-value: {pval:.6f} {'(유의미함 ✓)' if pval < 0.05 else '(유의미하지 않음 ✗)'}")
    print(f"  추정 효과: 평시 대비 {pct_effect:+.2f}% 매출 변화")
    print(f"  95% 신뢰구간: [{ci_low:+.2f}%, {ci_high:+.2f}%]")
    
    return {'coef': coef, 'p_value': pval, 'effect_pct': pct_effect}

def validate_overfitting(y, X, m=7, test_ratio=0.2):
    """Step 9: Train/Test 분할을 통한 모델 과적합 진단 및 시각화"""
    total_size = len(y)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size

    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    
    print(f"\n【Train/Test 분할】")
    print(f"  전체 데이터 크기: {total_size}")
    print(f"  학습(80%): {len(y_train)}일 ({y_train.index.min().date()} ~ {y_train.index.max().date()})")
    print(f"  테스트(20%): {len(y_test)}일 ({y_test.index.min().date()} ~ {y_test.index.max().date()})")

    print(f"\n--- 과적합 진단용 재학습 (Train: {len(y_train)}일, Test: {len(y_test)}일) ---")
    
    # Train 데이터로 모델 학습
    model_train_fit = pm.auto_arima(
        y_train,            # 훈련 시계열
        X=X_train,          # 훈련 외생변수
        m=m,
        d=None, D=None,
        max_p=2, max_q=2,
        max_P=2, max_Q=2,
        stepwise=True,
        trace=True,         # 과정 출력
        suppress_warnings=True
    )

    print("훈련 세트 auto_arima 학습 완료.")
    print(f"훈련 세트가 찾은 최적 order: {model_train_fit.order}, {model_train_fit.seasonal_order}")

    # 학습 예측
    train_pred = model_train_fit.predict_in_sample(X=X_train)
    train_resid = y_train - train_pred

    train_rmse = np.sqrt((train_resid ** 2).mean())
    train_mape = (np.abs(train_resid / y_train).mean()) * 100

    print(f"\n【학습 데이터 성능】")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAPE: {train_mape:.2f}%")

    # ========================================
    # 9-3) 테스트 데이터 예측
    # ========================================
    pred_result = model_train_fit.predict(
        n_periods=test_size,
        X=X_test,
        return_conf_int=True,   # 신뢰구간 요청
        alpha=0.05              # 95% 신뢰구간
    )

    test_pred = pred_result[0]
    conf_int_array = pred_result[1]

    # np.ndarray를 DataFrame으로 변환
    test_pred_ci = pd.DataFrame(
        conf_int_array,
        columns=['lower_ci', 'upper_ci'],
        index=y_test.index
    )

    # 인덱스 리셋 후 오차 계산
    test_pred.index = y_test.index

    test_resid = y_test - test_pred
    test_rmse = np.sqrt((test_resid ** 2).mean())
    test_mape = (np.abs(test_resid / y_test).mean()) * 100

    print(f"\n【테스트 데이터 성능】")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")

    # ========================================
    # 9-4) 과적합 진단
    # ========================================
    print(f"\n【과적합 진단】")

    rmse_ratio = test_rmse / train_rmse
    mape_ratio = test_mape / train_mape

    print(f"  RMSE 비율 (Test/Train): {rmse_ratio:.2f}")
    print(f"  MAPE 비율 (Test/Train): {mape_ratio:.2f}")

    if rmse_ratio > 1.3 or mape_ratio > 1.3:
        print(f"  ⚠️ 과적합 의심 (오류가 30% 이상 증가)")
    elif rmse_ratio > 1.1 or mape_ratio > 1.1:
        print(f"  ⚠️ 약간의 과적합 (정상 범위 상한)")
    else:
        print(f"  ✓ 과적합 없음 (모형 안정적)")

    # ========================================
    # 9-5) 예측 시각화
    # ========================================
    fig, ax = plt.subplots(figsize=(15, 6))

    # 학습 데이터
    ax.plot(y_train.index, y_train, label='학습 데이터', color='#11224E', alpha=0.7)
    ax.plot(train_pred.index, train_pred, label='학습 예측', color='#F87B1B', linestyle='--', linewidth=1)

    # 테스트 데이터
    ax.plot(y_test.index, y_test, label='테스트 데이터', color='green', alpha=0.7)
    ax.plot(test_pred.index, test_pred, label='테스트 예측', color='red', linestyle='--', linewidth=2)
    ax.fill_between(test_pred_ci.index, test_pred_ci['lower_ci'], test_pred_ci['upper_ci'],
                    alpha=0.2, color='red', label='95% CI')

    ax.set_xlabel('날짜', fontsize=14)
    ax.set_ylabel('Log 매출액', fontsize=14)
    ax.set_title('SARIMAX 예측 vs 실제', fontsize=18)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()