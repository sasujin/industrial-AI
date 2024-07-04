from durable.lang import *

# 규칙 정의
with ruleset('device_support'):
    # Windows 운영체제를 지원하는 경우
    @when_all(m.os == 'Windows')
    def support_windows(c):
        print('This device supports Windows OS')

    # MacOS 운영체제를 지원하는 경우
    @when_all(m.os == 'MacOS')
    def support_macos(c):
        print('This device supports MacOS')

    # Linux 운영체제를 지원하는 경우
    @when_all(m.os == 'Linux')
    def support_linux(c):
        print('This device supports Linux')

    # 안드로이드 운영체제를 지원하는 경우
    @when_all(m.os == 'Android')
    def support_android(c):
        print('This device supports Android OS')

    # iOS 운영체제를 지원하는 경우
    @when_all(m.os == 'iOS')
    def support_ios(c):
        print('This device supports iOS')

# 규칙 실행
post('device_support', {'os': 'Windows'})
post('device_support', {'os': 'MacOS'})
post('device_support', {'os': 'Linux'})
post('device_support', {'os': 'Android'})
post('device_support', {'os': 'iOS'})
