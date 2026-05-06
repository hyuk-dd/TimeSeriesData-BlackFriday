import pandas as pd
import glob



# 데이터 처리 함수화
def process_city_data(city_name, category_name, target_year):
    """
    주어진 도시, 업종 대분류, 단일 연도에 해당하는 데이터를 로드, 전처리, 집계하여 csv로 저장
    Args:
        city_name (str): 처리할 도시 이름 (예: '수원시')
        category_name (str): 필터링할 업종 대분류 이름 (예: '소매/유통')
        target_year (int): 처리할 연도 (예: 2024)
    """

    year_str = str(target_year)
    print(f"--- 🏙️ {city_name} ({year_str}) / 🛒 {category_name} 데이터 처리 시작 ---")

    df_list = []
    month_pattern = '[0-1][0-9]'    # 01월~12월

    # 1. 파일 로드 및 결합 --------------------------------------------------------

    folder_pattern = f"카드소비 데이터_{year_str}{month_pattern}"
    ptn_simple = f"{folder_pattern}/tbsh_gyeonggi_day_{city_name}.csv"
    ptn_monthly_1 = f"{folder_pattern}/tbsh_gyeonggi_day_{year_str}{month_pattern}_{city_name}.csv"
    ptn_monthly_2 = f"{folder_pattern}/tbsh_gyeonggi_day_{city_name}_{year_str}{month_pattern}.csv"

    # glob.glob를 사용하여 파일 목록을 가져오는 함수
    def get_files(pattern_list):
        files = []
        for ptn in pattern_list:
            files.extend(glob.glob(ptn))
        return list(set(files)) # 중복 파일 경로 제거

    # 찾을 모든 패턴 목록
    all_patterns = [ptn_simple, ptn_monthly_1, ptn_monthly_2]
    file_list = get_files(all_patterns)

    if not file_list:
        print(f"❌ 경고: {city_name}의 {year_str}년도 데이터 파일 및 통합 파일이 발견되지 않았습니다. 처리를 중단합니다.")
        return

    # 파일 로드
    for fname in file_list:
        try:
            temp = pd.read_csv(fname, encoding='utf8')
        except UnicodeDecodeError:
            temp = pd.read_csv(fname, encoding='cp949')
        df_list.append(temp)
        
    raw_data = pd.concat(df_list, ignore_index=True)
    # 날짜, 업종코드 기준 오름차순 정렬
    raw_data.sort_values(
    by=['ta_ymd', 'card_tpbuz_cd'],
    ascending=[True, True],
    ignore_index=True,
    inplace=True
    )
    print(f"로드 완료. 원본 데이터 크기: {raw_data.shape}")


    # 2. 데이터 전처리 --------------------------------------------------------

    # 기준연월일(날짜) 정수형이면 datetime으로 타입 변경
    try:
        raw_data['ta_ymd'] = pd.to_datetime(raw_data['ta_ymd'], format='%Y%m%d')
    except ValueError:
        pass

    # 업종 대분류 필터링
    filtered_data = raw_data.loc[raw_data['card_tpbuz_nm_1'] == category_name]

    # 윤년 데이터 삭제 (2월 29일)
    # 현재 연도 윤년인지 확인
    is_leap = (target_year % 4 == 0 and target_year % 100 != 0) or (target_year % 400 == 0)
    if is_leap:
        target_date = pd.to_datetime(f'{year_str}-02-29')
        # 2월 29일과 같지 않은 행만 남김
        filtered_data = filtered_data[filtered_data['ta_ymd'] != target_date]
        print(f"   -> {year_str}년 2월 29일 데이터 제거 완료")

    print(f"필터링(업종/윤년제거) 후 행 수: {len(filtered_data)}")

    # 불필요한 열 제거
    columns_to_drop = ['cty_rgn_no', 'admi_cty_no', 'card_tpbuz_nm_1', 'hour']
    cols_exist = [col for col in columns_to_drop if col in filtered_data.columns]
    df = filtered_data.drop(columns=cols_exist).copy()


    # 3. 데이터 집계 --------------------------------------------------------

    # 동일한 행 합치기
    # 그룹핑 키: 날짜, 업종분류코드, 업종중분류명, 성별, 나이, 요일
    # as_index: 그룹화 기준이 된 열을 결과 DF의 인덱스로 사용할지 여부
    group_keys = ['ta_ymd', 'card_tpbuz_cd', 'card_tpbuz_nm_2', 'sex', 'age', 'day']
    existing_keys = [key for key in group_keys if key in df.columns]

    final_df = df.groupby(
        existing_keys, as_index=False
    ).agg({'amt': 'sum', 'cnt': 'sum'})

    print(f"집계 후 행 수: {len(final_df)}")


    # 4. CSV 파일로 저장 --------------------------------------------------------
    output_filename = f"{category_name.replace('/', '_')}_{year_str}_{city_name}.csv"
    final_df.to_csv(
        output_filename, 
        index=False, 
        encoding='cp949'
    )

    print(f"✅ 최종 데이터가 '{output_filename}'로 저장되었습니다.")
    return final_df


### 데이터 로드
suwon_2022 = process_city_data('수원시', '소매/유통', 2022)
suwon_2023 = process_city_data('수원시', '소매/유통', 2023)
suwon_2024 = process_city_data('수원시', '소매/유통', 2024)
# suwon_2025 = process_city_data('수원시', '소매/유통', 2025)

yongin_2022 = process_city_data('용인시', '소매/유통', 2022)
yongin_2023 = process_city_data('용인시', '소매/유통', 2023)
yongin_2024 = process_city_data('용인시', '소매/유통', 2024)
# yongin_2025 = process_city_data('용인시', '소매/유통', 2025)

hwaseong_2022 = process_city_data('화성시', '소매/유통', 2022)
hwaseong_2023 = process_city_data('화성시', '소매/유통', 2023)
hwaseong_2024 = process_city_data('화성시', '소매/유통', 2024)
# hwaseong_2025 = process_city_data('화성시', '소매/유통', 2025)
