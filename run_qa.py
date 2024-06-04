import os
import openai
from openai import OpenAI

import pandas as pd

# recommand first way to set the private api key for safety
openai.api_key = os.environ.get("OPENAI_API_KEY")
# openai.api_key = "sk-****-*****"

# you can modified this prompt to test and improve the performance of GPT model
prompt_simple = """
    你是一个人工智能助手，现在需要完成一个文档检索问答任务。目标文档均来自于http://paper.people.com.cn/rmrb，你需要从2023年5月到2024年4月的新闻中检索相关信息。由于
    信息限制，我无法为你提供具体的文档链接，需要你自己检索。具体请参照以下示例， 其中Q代表问题，A代表标准回答，url代表文档链接：
        1. url: http://paper.people.com.cn/rmrb/html/2024-05/14/nw.D110000renmrb_20240514_1-01.htm#/
           Q: 2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？
           A: 湖南第一师范学院
        2. url: http://paper.people.com.cn/rmrb/html/2024-05/09/nw.D110000renmrb_20240509_1-06.htm#/
           Q: 2024年是中国红十字会成立多少周年
           A: 120
        3. url: http://paper.people.com.cn/rmrb/html/2024-03/12/nw.D110000renmrb_20240312_1-06.htm
           Q: 2024年我国文化和旅游部部长是谁？
           A: 孙业礼
        4. url: http://paper.people.com.cn/rmrb/html/2024-02/15/nw.D110000renmrb_20240215_1-02.htm
           Q: 《中华人民共和国爱国主义教育法》什么时候实施？
           A: 2024年1月1日
        5. url: http://paper.people.com.cn/rmrb/html/2023-12/12/nw.D110000renmrb_20231212_1-15.htm
           Q: 2023—2024赛季国际滑联短道速滑世界杯北京站比赛中，刘少昂参与获得几枚奖牌？
           A: 2
        6. url: http://paper.people.com.cn/rmrb/html/2023-11/03/nw.D110000renmrb_20231103_1-06.htm
           Q: 福建自贸试验区在自贸建设十年中主要从哪几个方面推动改革创新？
           A: 推进制度集成创新；服务海峡两岸融合发展；深化共建“一带一路” 
        7. url: http://paper.people.com.cn/rmrb/html/2023-10/09/nw.D110000renmrb_20231009_1-10.htm
           Q: 杭州第十九届亚洲运动会共举行多少天？
           A: 16
        8. url: http://paper.people.com.cn/rmrb/html/2024-04/09/nw.D110000renmrb_20240409_4-14.htm
           Q: 2023年广西植树造林面积大约多少亩？
           A: 417万亩
    请使用最简短的答案回答以下问题，请不要将文档的网址链接放入答案中，并去掉所有答案的标点符号。
"""

prompt_open = """
    你是一个人工智能助手，现在需要完成一个文档检索问答任务。目标文档均来自于http://paper.people.com.cn/rmrb，你需要从2023年5月到2024年4月的新闻中检索相关信息。由于
    信息限制，我无法为你提供具体的文档链接，需要你自己检索。具体请参照以下示例， 其中Q代表问题，A代表标准回答，url代表文档链接：
        1. url: http://paper.people.com.cn/rmrb/html/2024-05/14/nw.D110000renmrb_20240514_1-01.htm#/
           Q: 2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？
           A: 湖南第一师范学院
        2. url: http://paper.people.com.cn/rmrb/html/2024-05/09/nw.D110000renmrb_20240509_1-06.htm#/
           Q: 2024年是中国红十字会成立多少周年
           A: 120
        3. url: http://paper.people.com.cn/rmrb/html/2024-03/12/nw.D110000renmrb_20240312_1-06.htm
           Q: 2024年我国文化和旅游部部长是谁？
           A: 孙业礼
        4. url: http://paper.people.com.cn/rmrb/html/2024-02/15/nw.D110000renmrb_20240215_1-02.htm
           Q: 《中华人民共和国爱国主义教育法》什么时候实施？
           A: 2024年1月1日
        5. url: http://paper.people.com.cn/rmrb/html/2023-12/12/nw.D110000renmrb_20231212_1-15.htm
           Q: 2023—2024赛季国际滑联短道速滑世界杯北京站比赛中，刘少昂参与获得几枚奖牌？
           A: 2
        6. url: http://paper.people.com.cn/rmrb/html/2023-11/03/nw.D110000renmrb_20231103_1-06.htm
           Q: 福建自贸试验区在自贸建设十年中主要从哪几个方面推动改革创新？
           A: 推进制度集成创新；服务海峡两岸融合发展；深化共建“一带一路” 
        7. url: http://paper.people.com.cn/rmrb/html/2023-10/09/nw.D110000renmrb_20231009_1-10.htm
           Q: 杭州第十九届亚洲运动会共举行多少天？
           A: 16
        8. url: http://paper.people.com.cn/rmrb/html/2024-04/09/nw.D110000renmrb_20240409_4-14.htm
           Q: 2023年广西植树造林面积大约多少亩？
           A: 417万亩
    请使用长度不超过512的答案回答以下问题，可以使用至多5个网页文档作为参考依据并将全部使用的文档的网址链接放入答案中，使用换行符分隔答案文本与网址链接。
"""


def read_tsv(file_path):
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8', header=None)
    return df


def data_preprocessing(queries):
    """
    :param: queires: df.DataFrame
    :return: questions: List
             ground truth answers: List
    """

    q = []
    gt = []

    for _, data in queries.iterrows():
        try:
            q.append(data.iloc[0].split("|||")[0])
            gt.append(data.iloc[0].split("|||")[1])
        except:
            continue
    return q, gt


def get_answers(query, prompt):
    client = OpenAI()

    answer = []
    urls = []

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"{prompt}\nQ：{query}"}],
        stream=True,
    )
    for chunk in stream:
        # print(chunk.choices[0].delta.content or "", end="")
        prediction = chunk.choices[0].delta.content or ""
        answer.append(prediction)
        #answer.append("".join(prediction).split("\n")[0])
        #urls.append("".join(prediction).split("\n")[1])
    return "".join(answer), urls


def evaluate(queries: list):
    """
    queries: List[str] 输入查询列表
    Return: List[str] 输出答案列表
    """

    predictions = []
    urls = []

    # you can add question type and add elif sentence to assign question to different prompts
    q_type = input("Please select your quetion type from following tuple: (simple, open)\n")

    while q_type not in ["simple", "open"]:
            q_type = input("Please only select your quetion type from following tuple: (simple, open)\n")

    if q_type.lower() == "simple":
        prompt = prompt_simple
    elif q_type.lower() == "open":
        prompt = prompt_open
    else:
        pass
    
    for q in queries:
        p, url = get_answers(q, prompt)
        predictions.append(p)
        urls.append(url)

    return predictions


if __name__=="__main__":
    queries = read_tsv("data/example_ans.tsv")
    open_question = pd.read_csv("data/开放题.txt", header=None)
    
    q, gt = data_preprocessing(queries)
    q_open = [q.iloc[0] for _, q in open_question.iterrows()]
    
    predictions = evaluate(q)

    print(predictions)

    predictions_open = evaluate(q_open)

    print(predictions_open)
