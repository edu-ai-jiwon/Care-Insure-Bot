"""
TRICARE 보험 제외 항목 크롤링
- 메인 페이지: https://www.tricare.mil/CoveredServices/IsItCovered/Exclusions
- 각 제외 항목 링크를 순회하며 상세 내용 수집
- 결과: CSV 저장 + LangChain Document 리스트 반환
"""

import time
import csv
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document



# 공통 설정
# tricare.mil이 봇 차단을 하기 때문에 브라우저처럼 보이게 헤더 설정
BASE_URL = 'https://www.tricare.mil'
EXCLUSIONS_URL = f'{BASE_URL}/CoveredServices/IsItCovered/Exclusions'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}



# 메인 페이지에서 제외 항목 링크 목록 수집
# <ul> 안의 <li><a href="/CoveredServices/..."> 구조를 파싱
def get_exclusion_links(url):
    """
    제외 항목 메인 페이지에서 각 항목의 이름과 URL을 수집합니다.
    Returns:
        list of dict: [{'name': 'Acupuncture', 'url': 'https://...'}, ...]
    """
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # 개발자 도구 기준: div.col.col-1 안의 ul > li > a
    col1 = soup.find('div', class_='col-1')
    links = []

    if col1:
        for a_tag in col1.find_all('a', href=True):
            href = a_tag['href']
            name = a_tag.get_text(strip=True)

            # /CoveredServices/IsItCovered/ 경로만 수집
            if '/CoveredServices/IsItCovered/' in href and href != '/CoveredServices/IsItCovered/Exclusions':
                links.append({
                    'name': name,
                    'url': BASE_URL + href if href.startswith('/') else href
                })

    print(f'✅ 제외 항목 링크 {len(links)}개 수집 완료')
    return links



# 각 제외 항목 페이지에서 상세 내용 추출
# 개발자 도구 기으로함\
# div#content.col.col-1 안의 h1 + p 태그
def get_exclusion_detail(name, url):
    """
    제외 항목 상세 페이지에서 제목과 본문을 추출
    Returns:
        dict: {'name': ..., 'url': ..., 'content': ...}
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # 개발자 도구 기준: div#content 안에 h1과 p가 있음
        content_div = soup.find('div', id='content')
        if not content_div:
            # fallback: col-1 클래스로 시도
            content_div = soup.find('div', class_='col-1')

        if not content_div:
            return {'name': name, 'url': url, 'content': '내용 없음'}

        # h1 제목 추출
        h1 = content_div.find('h1')
        title = h1.get_text(strip=True) if h1 else name

        # p 태그 본문 추출 (여러 개 있을 수 있음)
        paragraphs = content_div.find_all('p')
        body = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        return {
            'name': name,
            'url': url,
            'title': title,
            'content': body
        }

    except Exception as e:
        print(f'  ⚠️  {name} 수집 실패: {e}')
        return {'name': name, 'url': url, 'title': name, 'content': '수집 실패'}



# 4단계 — 전체 크롤링 실행
# 요청 사이에 1초 딜레이 
# 서버 부하 방지 + 차단 방지
def crawl_exclusions():
    """
    제외 항목 전체를 크롤링해서 상세 내용 리스트를 반환
    Returns:
        list of dict: [{'name': ..., 'url': ..., 'title': ..., 'content': ...}, ...]
    """
    links = get_exclusion_links(EXCLUSIONS_URL)
    results = []

    for i, item in enumerate(links):
        print(f'  [{i+1}/{len(links)}] {item["name"]} 수집 중...')
        detail = get_exclusion_detail(item['name'], item['url'])
        results.append(detail)
        time.sleep(1)  # 서버 부하 방지

    print(f'\n✅ 총 {len(results)}개 항목 수집 완료')
    return results



# 5단계 — CSV로 저장 (raw 데이터)
# 나중에 pandas로 불러오거나 직접 확인할 때 사용
def save_to_csv(results, filepath='tricare_exclusions.csv'):
    """
    크롤링 결과를 CSV 파일로 저장합니다.
    """
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'title', 'content', 'url'])
        writer.writeheader()
        writer.writerows(results)

    print(f'✅ CSV 저장 완료: {filepath}')



# LangChain Document로 변환 (RAG 데이터)
# CSV 변환 단계와 동일한 방식:
# - page_content에 항목명 포함: 청킹 후에도 컨텍스트 유지
# - metadata에 출처 정보 저장: 검색 결과 추적 가능
def to_documents(results):
    """
    크롤링 결과를 LangChain Document 리스트로 변환합니다.
    """
    docs = []
    for item in results:
        doc = Document(
            page_content=(
                f"제외 항목: {item['title']}\n"
                f"내용: {item['content']}"
            ),
            metadata={
                'source': item['url'],
                '항목명': item['name'],
                'type': 'exclusion'   # 기존 mental_health와 구분
            }
        )
        docs.append(doc)

    print(f'✅ Document 변환 완료: {len(docs)}개')
    return docs



# 실행
if __name__ == '__main__':
    # 1. 크롤링
    results = crawl_exclusions()

    # 2. CSV 저장 (raw 데이터)
    save_to_csv(results, 'tricare_exclusions.csv')

    # 3. Document 변환 (RAG 데이터)
    exclusion_docs = to_documents(results)

    # 4. 샘플 확인
    print('\n--- 샘플 ---')
    print(exclusion_docs[0].page_content[:300])
    print(f'\nMetadata: {exclusion_docs[0].metadata}')
