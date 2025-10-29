import pandas as pd  # 데이터 처리 및 분석
import re  # 정규 표현식


# 텍스트 전처리 파이프라인 클래스 구성
class TextPreprocessingPipeline:
    """
    텍스트 전처리 파이프라인 클래스
    - 기본 전처리와 학습 데이터 기반 고급 전처리를 통합 관리
    - 재사용 가능하고 확장 가능한 구조
    """

    def __init__(self):
        self.is_fitted = False
        self.vocab_info = {}
        self.label_patterns = {}

    def basic_preprocess(self, texts):
        """기본 전처리 (clean_text + normalize 기능)"""
        processed_texts = []
        for text in texts:
            # 기본 텍스트 정리
            cleaned = self._clean_text(text)
            # 정규화
            normalized = self._normalize_text(cleaned)
            processed_texts.append(normalized)
        return processed_texts

    def _clean_text(self, text):
        """기존 clean_text 함수 내용"""
        if pd.isna(text):
            return ""

        text = str(text).strip()
        text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", "", text)
        text = re.sub(r"([ㅋㅎ])\1{2,}", r"\1\1", text)
        text = re.sub(r"([ㅠㅜㅡ])\1{2,}", r"\1\1", text)
        text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
        text = re.sub(r"[^\w\s가-힣.,!?ㅋㅎㅠㅜㅡ~\-]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _normalize_text(self, text):
        """텍스트 정규화 함수"""
        # 소문자 변환
        text = text.lower()

        # 구두점 정규화
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[,]{2,}", ",", text)
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"([.,!?])\s+", r"\1 ", text)

        # 특수문자 정리
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def fit(self, texts, labels=None):
        """학습 데이터로부터 전처리 정보 학습"""
        # 구현 X
        self.is_fitted = True

    def transform(self, texts, labels=None):
        """전처리 적용"""
        if not self.is_fitted:
            print(
                "Warning: 파이프라인이 학습되지 않았습니다. 기본 전처리만 적용합니다."
            )
            return self.basic_preprocess(texts)
            
        # 1. clean_text + normalize
        processed_texts_list = self.basic_preprocess(texts)
        
        if labels is None:
            df_text = pd.DataFrame({'review_cleaned': processed_texts_list})
        else:
            df_text = pd.DataFrame({'review_cleaned': processed_texts_list, 'label': labels})

        # 2. 중복 제거
        initial_count = len(df_text)
        if labels is None:
            df_text.drop_duplicates(subset=["review_cleaned"], inplace=True)
        else:
            df_text.drop_duplicates(subset=["review_cleaned", "label"], inplace=True)
        duplicates_count = initial_count - len(df_text)
        if duplicates_count > 0:
            print(f"[Fit/Transform] 중복 데이터 {duplicates_count}개 제거")
            
        # 3. 빈 텍스트 제거 (공백만 남은 텍스트 제거)
        initial_count = len(df_text)
        df_text = df_text[df_text["review_cleaned"].str.strip().str.len() > 0]
        empty_count = initial_count - len(df_text)
        if empty_count > 0:
            print(f"[Fit/Transform] 빈 텍스트 {empty_count}개 제거")

        # 4. 최종적으로 정제된 텍스트 목록(X_train)과 레이블(y_train) 반환
        if labels is None:
            return df_text["review_cleaned"]
        else:
            return df_text["review_cleaned"], df_text["label"]


    def fit_transform(self, texts, labels=None):
        """학습과 변환을 동시에 수행"""
        self.fit(texts, labels)
        return self.transform(texts, labels)


# 전처리 파이프라인 인스턴스 생성
preprocessor = TextPreprocessingPipeline()