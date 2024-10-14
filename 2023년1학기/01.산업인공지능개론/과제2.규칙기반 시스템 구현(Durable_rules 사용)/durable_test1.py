
from durable.lang import *
from datetime import datetime, timedelta

with ruleset('personal_information_rules'):
    # 개인정보 유출 이벤트 발생 시, 대응 방안 제시
    @when_all((m.event == 'personal_information_leak') & (m.severity == 'high'))
    def suggest_response(c):
        data_type = c.m.data_type
        amount = c.m.amount
        leak_location = c.m.leak_location
        time_diff = datetime.now() - datetime.strptime(c.m.time, '%Y-%m-%d %H:%M:%S')
        # 신용카드 예시
        if data_type == 'credit_card':
            if amount > 1000:
                if leak_location == 'internal':
                    print('관리자에게 경고 이메일 보내기')
                    print('사용자 계정 잠금')
                    print('--------------------------------------------------------')
                elif leak_location == 'external':
                    print('관리자에게 경고 이메일 보내기')
                    print('동결 신용 카드')
                    print('사용자 위치 외부에서 신용 카드 사용 불가')
                    print('--------------------------------------------------------')                    
            else:
                print('영향을 받는 사용자에게 경고 이메일 보내기')
                print('--------------------------------------------------------')                
        # 주민등록번호 예시
        elif data_type == 'social_security_number':
            if leak_location == 'internal':
                print('관리자에게 경고 이메일 보내기')
                print('사용자 계정 잠금')
                print('관련 기관에 보고')
                print('--------------------------------------------------------')                
            elif leak_location == 'external':
                print('영향을 받는 사용자에게 경고 이메일 보내기')
                print('신용 모니터링 서비스 제공')
                print('관련 기관에 보고')
                print('--------------------------------------------------------')
        # 유출된 시간이 1시간 이내인 경우, 추가 대응 방안 제시
        if time_diff <= timedelta(hours=1):
            if data_type == 'credit_card':
                print('1시간 이내인 경우 동결 신용 카드')
                print('--------------------------------------------------------')                
            elif data_type == 'social_security_number':
                print('1시간 이내인 경우 신용 모니터링 서비스 제공')
                print('--------------------------------------------------------')                
            else:
                print('조치 필요 없음')
                print('--------------------------------------------------------')                

# 규칙 실행
post('personal_information_rules', {'event': 'personal_information_leak', 'severity': 'high', 'data_type': 'credit_card', 'amount': 2000, 'leak_location': 'internal', 'time': '2023-04-17 15:30:00'})
post('personal_information_rules', {'event': 'personal_information_leak', 'severity': 'high', 'data_type': 'credit_card', 'amount': 500, 'leak_location': 'external', 'time': '2023-04-17 14:30:00'})
post('personal_information_rules', {'event': 'personal_information_leak', 'severity': 'high', 'data_type': 'social_security_number', 'leak_location': 'internal', 'time': '2023-04-17 15:00:00'})
post('personal_information_rules', {'event': 'personal_information_leak', 'severity': 'high', 'data_type': 'social_security_number', 'leak_location': 'external', 'time': '2023-04-17 13:00:00'})