# 1. Load Data
import pandas as pd

# input
file_path = 'train.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print("'utf-8' 인코딩으로 Load 성공")

except UnicodeDecodeError:
    print("'utf-8' 인코딩으로 실패, 'cp949' 인코딩으로 재시도")
    try:
        df = pd.read_csv(file_path, encoding='cp949')
        print("'cp949' 인코딩으로 Load 성공")
    except Exception as e:
        print(f"파일 로드 중 최종 오류 발생: {e}")
        # 오류가 발생하면 빈 DataFrame을 생성하고 종료
        df = pd.DataFrame()


if not df.empty:
    print("\n--- 데이터프레임 정보 ---")
    print(df.info())
    
    # print("\n--- 상위 5개 행 데이터 ---")
    # print(df.head())
    
    # review data만 출력하기
    pd.set_option('display.max_colwidth', None) 
    pd.set_option('display.max_rows', 100)
    second_column_100_rows = df.iloc[:100, 1]
    print(second_column_100_rows)
    
    '''
    데이터 전처리 계획
    1. raw 데이터다 보니 맞춤법이 틀린 경우가 많다.
    우선 맞춤법부터 교정한다.
    
    2. OoV 또한 많다고 생각한다.
    Subword tokenizatoin이 적합하다.
    
    3. pretrain 모델에서 이모티콘의 의미가 학습되었는지 의문이다.
    
    4. pretrain 모델마다 사용하는 전처리 방식이 다를 텐데, 전처리 방식을 스스로 생각하는 것이 의미가 있는지 의문이 든다.
    '''