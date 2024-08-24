from flask import Flask, render_template, request
from urllib.parse import quote, unquote
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 엑셀 데이터 로드
data = pd.read_excel('final_data.xlsx')

# 허용된 태그와 기술 스택 정의
allowed_tags = [
    '정보통신', '제조업', '의료/바이오', '에너지', '농업/식품', '금융/핀테크', '건설/건축', '환경', 
    '문화/예술', '교통', '교육', '관광/레저', '방송/미디어', '패션/뷰티', '자동차/모빌리티', '우주/항공', 
    '스포츠', '사회복지', '법률/정책', '화학', '해양/수산', '무역/국제', '공공안전', '게임/엔터테인먼트', 
    '반도체/전자', '도시계획/인프라'
]

allowed_tech_stack = [
    'Python', 'SQL', 'HTML/CSS', 'AI', 'Java', 'Data', 'R', 'Unity', 'AWS', 'TensorFlow', 
    'C++', 'Software', 'C#', 'Blockchain', 'Machine Learning', '3D', 'IoT', 'Engine', 'Unreal', 
    'Video', 'MongoDB', 'C'
]

# 데이터 전처리
data['Tags'] = data['Tags'].fillna('').apply(lambda x: ','.join([tag.strip() for tag in x.split(',') if tag.strip() in allowed_tags]))
data['Tech Stack'] = data['Tech Stack'].fillna('')

# 기술 스택 필터링 및 형식 유지
def format_tech_stack(stack):
    formatted_stack = []
    for tech in stack.split():
        tech = tech.strip()
        for allowed_tech in allowed_tech_stack:
            if tech.lower() == allowed_tech.lower():
                formatted_stack.append(allowed_tech)
                break
    return ','.join(formatted_stack)

data['Tech Stack'] = data['Tech Stack'].apply(format_tech_stack)

# 소개글 포맷팅 함수
def simple_format_description(description):
    # 줄바꿈을 <br> 태그로 대체하여 문단을 나눔
    return description.replace('\n', '<br>')

# 공모전 소개글을 포맷팅
data['공모전 소개'] = data['공모전 소개'].apply(simple_format_description)

# 디버깅: 기술 스택 필터링 후 출력
print("After filtering:", data['Tech Stack'].unique())

data['Combined'] = data['Tags'] + ' ' + data['Tech Stack']

# 허용된 태그와 기술 스택만 추출
unique_tags = sorted(set(tag.strip() for tags in data['Tags'].dropna() for tag in tags.split(',')))
unique_tech_stack = sorted(set(stack.strip() for stacks in data['Tech Stack'].dropna() for stack in stacks.split(',')))

# 디버깅: 최종 추출된 기술 스택 출력
print("Unique tech stacks:", unique_tech_stack)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Combined'])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 난이도를 🔥 모양으로 변환하기 위한 매핑
difficulty_mapping = {
    '매우 어려움': 5,
    '어려움': 4,
    '보통': 3,
    '쉬움': 2,
    '매우 쉬움': 1
}

def convert_difficulty(difficulty):
    # 난이도 문자열을 숫자에 매핑하고 🔥 모양으로 변환
    level = difficulty_mapping.get(difficulty, 0)
    return '🔥' * level

data['난이도_표시'] = data['난이도'].apply(convert_difficulty)

# D-day 형식 변환
def format_d_day(d_day):
    if isinstance(d_day, str):
        # 문자열 값이 "마감"인 경우 그대로 반환
        if d_day.lower() == '마감':
            return '마감'
        else:
            return 'D-day'
    elif isinstance(d_day, (int, float)):
        # 숫자 값인 경우 D-day 형식으로 반환
        if d_day == 0:
            return '마감'
        elif d_day > 0:
            return f'D-{int(d_day)}'
        else:
            return 'D-day'
    else:
        # 예상치 못한 값이 있는 경우 기본값 반환
        return 'D-day'

data['D-day_표시'] = data['D-day'].apply(format_d_day)

@app.route('/')
def index():
    contests = data[['공모전명', '기관명', '기간', '1등시상금', '이미지url', 'Tags', 'Tech Stack', 'D-day_표시', '총시상금', '난이도_표시']].to_dict(orient='records')
    return render_template('index.html', contests=contests, unique_tags=unique_tags, unique_tech_stack=unique_tech_stack)

# 공모전 상세 페이지 및 유사 공모전 추천
@app.route('/contest/<path:name>')
def contest_detail(name):
    name = unquote(name)  # URL에서 받은 name을 디코딩
    if name not in data['공모전명'].values:
        return "공모전 정보를 찾을 수 없습니다.", 404
    contest = data[data['공모전명'] == name].iloc[0]
    similar_contests = find_similar_contests(name)
    return render_template('detail.html', contest=contest, similar_contests=similar_contests)

def get_top_similar(idx, n=5):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores[1:n+1]

def find_similar_contests(contest_name, n=5):
    if contest_name not in data['공모전명'].values:
        return None
    
    contest_idx = data.index[data['공모전명'] == contest_name].tolist()[0]
    similar_indices = get_top_similar(contest_idx, n=n)
    
    similar_contests = []
    for idx, score in similar_indices:
        if idx != contest_idx:
            similar_contests.append({
                '공모전명': data.loc[idx, '공모전명'],
                '기관명': data.loc[idx, '기관명'],
                '이미지url': data.loc[idx, '이미지url'],
                '유사도': score
            })
    return similar_contests

if __name__ == '__main__':
    app.run(debug=True)
