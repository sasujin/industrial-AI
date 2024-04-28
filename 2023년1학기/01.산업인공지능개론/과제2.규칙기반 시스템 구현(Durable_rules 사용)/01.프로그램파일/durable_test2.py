from durable.lang import *

# 버전 정보
version_info = {
    'service1': '1.2.3',
    'service2': '4.5.6',
    'service3': '7.8.9'
}

# 규칙 정의
with ruleset('version_rules'):
    
    # 서비스 버전이 일치하지 않는 경우
    @when_all((m.service1.version != version_info['service1']) |
              (m.service2.version != version_info['service2']) |
              (m.service3.version != version_info['service3']))
    def mismatched_version(c):
        print('서비스 버전이 일치하지 않습니다.')
        print('현재 버전: ')
        print(f"서비스 1: {m.service1.version}")
        print(f"서비스 2: {m.service2.version}")
        print(f"서비스 3: {m.service3.version}")
        
    # 서비스 버전이 일치하는 경우
    @when_all((m.service1.version == version_info['service1']) &
              (m.service2.version == version_info['service2']) &
              (m.service3.version == version_info['service3']))
    def matched_version(c):
        print('서비스 버전이 모두 일치합니다.')
        
# 규칙 실행
post('version_rules', {'service1': {'version': '1.2.3'},
                       'service2': {'version': '4.5.6'},
                       'service3': {'version': '7.8.9'}})
