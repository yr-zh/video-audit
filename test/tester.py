import requests
import base64
import json

# 服务器地址
URL = "http://127.0.0.1:80/v1/multiple_classification/predict"

# 读取文件并进行 Base64 编码
def encode_file(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# 解析 test_data
test_data = {
    "id": "3133",
    "label": "Q0",
    "ctype": "mix",
    "content": {
        "meta": {
            "type": "raw",
            "value": {
                "itemId": "3133",
                "content": "见证2024巴黎奥运会上的震撼对决！澳大利亚34-5南非，女子七人制橄榄球，池B组激战。Madison Levi，这位在东京崭露头角的巨星，再次以惊人的速度和技巧，引领澳大利亚队横扫对手。精彩瞬间不断，从精准传球到闪电得分，每一分都充满激情与策略。不容错过的体育盛事，带你领略女子橄榄球的无限魅力！",
                "itemTime": 1723094562361,
                "title": "Australia 34-5 South Africa  Women's Pool B  Rugby Sevens  Olympic Games Paris 2024",
                "url": "https://material-api.iwhopodm.com/res/video/588987_4d3fa864f90739294620.mp4",
                "duration": 482,
                "categoryLevel1": "体育",
                "tag": "澳洲, 橄榄球, 女子, 七人制, 波B",
                "coverUrl": "https://cn-material-bucket.oss-cn-shenzhen.aliyuncs.com/res/img/115764_426ca6d8dc716f90d265.jpg"
            }
        },
        "cover": {"type": "file", "value": "/tmp/dataset/covers/ef/115764_426ca6d8dc716f90d265.jpg"},
        "video": {"type": "file", "value": "/tmp/dataset/videos/d3/588987_4d3fa864f90739294620.mp4"}
    }
}

# 获取文件路径
cover_filename = test_data["content"]["cover"]["value"].replace("/tmp/dataset/", "/mnt/data/leaderboard/56da1986c432e0fff72fb039491c3548/")
video_filename = test_data["content"]["video"]["value"].replace("/tmp/dataset/", "/mnt/data/leaderboard/56da1986c432e0fff72fb039491c3548/")

# 构造请求数据
payload = {
    "content": {
        "meta": test_data["content"]["meta"]["value"],
        "cover": encode_file(cover_filename),  # 图片 Base64
        "video": encode_file(video_filename)  # 视频 Base64
    }
}

# 设置请求头
headers = {
    "Content-Type": "application/json"
}

# 发送 POST 请求
response = requests.post(URL, headers=headers, data=json.dumps(payload))

# 输出响应结果
print("Status Code:", response.status_code)
print("Response:", response.json())
