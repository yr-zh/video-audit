# {"id": "335", "label": "Q2", "ctype": "mix", "content": {"meta": {"type": "raw", "value": {"itemId": "335", "content": "【2024奥运会】【女子10米气手枪】韩国选手吴艺真打破奥运会纪录拿下冠军", "itemTime": 1722339861392, "title": "【2024奥运会】【女子10米气手枪】韩国选手吴艺真打破奥运会纪录拿下冠军", "url": "https://cn-material-bucket.oss-cn-shenzhen.aliyuncs.com/res/video/50_0ef8aa4a3fc31bcd.mp4", "duration": 75, "categoryLevel1": "体育", "tag": "奥运会, 女子射击, 韩国选手, 10米气手枪, 冠军纪录", "coverUrl": "https://cn-material-bucket.oss-cn-shenzhen.aliyuncs.com/res/img/67_0ef8aa4a3fc31bcd.jpg", "bloggerName": null, "likeCnt": null, "commentCnt": null, "collectCnt": null, "fansCnt": null}}, "cover": {"type": "file", "value": "/tmp/dataset/covers/a4/67_0ef8aa4a3fc31bcd.jpg"}, "video": {"type": "file", "value": "/tmp/dataset/videos/1d/50_0ef8aa4a3fc31bcd.mp4"}}}

import os
import json 


# label_path = "index.jsonl"

# with open(label_path, 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines[:10]:
#         data = json.loads(line)
#         print(data)
#         label = data()


from torch.utils.data import Dataset

class CustomJSONLDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        初始化数据集
        :param file_path: JSONL 文件路径，每行是一个 JSON 对象
        :param transform: 可选的预处理或数据增强函数
        """
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取一条 JSON 数据
        item = self.data[idx]
        
        # 提取 label 与 ctype
        label = item.get("label", None)
        ctype = item.get("ctype", None)
        
        # 从 content 中提取关键信息
        content_text = None
        title = None
        tags = None
        if "content" in item and "meta" in item["content"]:
            meta_val = item["content"]["meta"].get("value", {})
            content_text = meta_val.get("content", None)
            title = meta_val.get("title", None)
            tags = meta_val.get("tag", None)
            content = item["content"]
        
        # 提取 cover 文件路径
        cover_path = None

        if "cover" in content and isinstance(content["cover"], dict):
            cover_path = content["cover"].get("value", None)

        # 提取 video 文件路径
        video_path = None
        if "video" in content and isinstance(content["video"], dict):
            video_path = content["video"].get("value", None)
        
        # 构造返回的字典
        result = {
            "id": item.get("id", None),
            "label": label,
            "ctype": ctype,
            "content": content_text,
            "title": title,
            "tag": tags,
            "cover": cover_path,
            "video": video_path
        }
        
        if self.transform:
            result = self.transform(result)
            
        return result

# 使用示例
if __name__ == "__main__":
    # dataset = CustomJSONLDataset("index.jsonl")
    # print("数据集大小:", len(dataset))
    # print("第一条数据:", dataset[0])

    # 把图片不存在的条目去除掉
    # file_path = "./datasets/fine/index.jsonl"
    # file_path = "./datasets/raw/index_train_auxiliary.jsonl"
    # dataset = []
    # with open(file_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         line = line.strip()
    #         if line:
    #             data = json.loads(line)
    #             cover = data["content"]["cover"]["value"]
    #             cover = cover.replace("/tmp/dataset/", "/mnt/data/leaderboard/56da1986c432e0fff72fb039491c3548/")
    #             if os.path.exists(cover):
    #                 dataset.append(line)
    #             else:
    #                 print(cover, " not exists")

    # with open("./datasets/raw/index_train_auxiliary_clean.jsonl", 'w', encoding="utf-8") as fw:
    #     fw.writelines(line + "\n" for line in dataset)

    count = 0
    file_path = "./datasets/raw/index_train_auxiliary_merge_20250410.jsonl"
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                cover = data["content"]["cover"]["value"]
                label = data["label"]
                if label in ["Q-1"]:
                    dataset.append(line)
                    count += 1

    with open("./datasets/raw/index_train_auxiliary_merge_20250410_Q-1.jsonl", 'w', encoding="utf-8") as fw:
        fw.writelines(line + "\n" for line in dataset)

    print(count)