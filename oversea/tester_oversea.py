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
# test_data = {"id": "5662898", "label": "Q0", "ctype": "mix", "content": {"meta": {"type": "raw", "value": {"itemId": "5662898", "content": "", "itemTime": 1743944913028, "title": "تقبلها أمام زوجها", "url": "https://hkcos-material.iwhop.cn/res/btu/video/youtube:LfDtSHQpcGgs_360_634.mp4", "duration": 5, "categoryLevel1": "美女", "tag": ",夫妻,冲突,谅解,对话,婚姻危机", "coverUrl": "https://hkcos-material.iwhop.cn/res/btu/video/youtube:LfDtSHQpcGgs_1280_720.jpg", "bloggerName": "btu_اجمل امراة في العالم ", "likeCnt": 35, "commentCnt": 0, "collectCnt": None, "fansCnt": 1300}}, "cover": {"type": "file", "value": "/tmp/dataset/covers/98/youtube:LfDtSHQpcGgs_1280_720.jpg"}, "video": {"type": "file", "value": "/tmp/dataset/videos/bf/youtube:LfDtSHQpcGgs_360_634.mp4"}}, "firstAuditor": "i-jiangxi@4paradigm.com", "firstAuditTime": 1744008999296, "secondAuditTime": 1744167636747, "firstQuality": "Q1"}
test_data = {"id": "5658358", "label": "Q2", "ctype": "mix", "content": {"meta": {"type": "raw", "value": {"itemId": "5658358", "content": "", "itemTime": 1743938443611, "title": "😲😲😲A woman survives a plane crash", "url": "https://hkcos-material.iwhop.cn/res/btu/video/youtube:2snVxkiBFh8s_360_640.mp4", "duration": 59, "categoryLevel1": "电影", "tag": "Plane crash, Survive, Woman, Accident, Miracle", "coverUrl": "https://hkcos-material.iwhop.cn/res/btu/video/youtube:2snVxkiBFh8s_1080_1920.jpg", "bloggerName": "ngyd_面包电影", "likeCnt": 168361, "commentCnt": 294, "collectCnt": None, "fansCnt": 215000}}, "cover": {"type": "file", "value": "/tmp/dataset/covers/9e/youtube:2snVxkiBFh8s_1080_1920.jpg"}, "video": {"type": "file", "value": "/tmp/dataset/videos/01/youtube:2snVxkiBFh8s_360_640.mp4"}}, "firstAuditor": "zhanghuanquan@4paradigm.com", "firstAuditTime": 1744098327573, "secondAuditTime": 1744167636656, "firstQuality": "Q2"}


# 获取文件路径
cover_filename = test_data["content"]["cover"]["value"].replace("/tmp/dataset/", "/mnt/data/leaderboard/32ca9448ba8c2b4e4409fd5edc5123a8/")
video_filename = test_data["content"]["video"]["value"].replace("/tmp/dataset/", "/mnt/data/leaderboard/32ca9448ba8c2b4e4409fd5edc5123a8/")

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
