import requests

if __name__ == '__main__':
    ### weixin token
    requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                         json={
                             "token": "69fcd3bf894d",
                             "title": "run status",
                             "name": "fail notice",
                             "content": "run failed!!!"
                         })