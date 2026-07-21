import json
import http.client
from openai import OpenAI
import requests
import os
from tqdm import tqdm

import fitz  # PyMuPDF

from drsr_420.tools.search_seper import search_google_scholar

# ── 客户端 ──────────────────────────────────────────────
deepseek = OpenAI(
    api_key="sk-3970c8c4922f49fd89761fe3ad4a5eb5",
    base_url="https://api.deepseek.com",        # 兼容 OpenAI 格式
)

SERPER_KEY = "ac28c1aac4d446f3de5c8e79ea6d406727509455"


# ── 工具 2：web visit ───────────────────────
def read_paper(pdf_url:str, title: str, save_dir="pdf_downloads") :
    """
    调 Serper 的 WebPage API，
    返回『已摘选结构化』的 JSON 字符串，方便模型消费。
    """
    """下载PDF文件并保存到本地（使用代理）"""
    # 代理配置
    proxyHost = "www.16yun.cn"
    proxyPort = "5445"
    proxyUser = "16QMSOML"
    proxyPass = "280651"

    # 构造代理字典
    proxies = {
        "http": f"http://{proxyUser}:{proxyPass}@{proxyHost}:{proxyPort}",
        "https": f"http://{proxyUser}:{proxyPass}@{proxyHost}:{proxyPort}"
    }

    # 请求头设置
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/pdf,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

    cookies ={
  "s_vi": "[CS]v1|3463D79773E0912A-400003FCC1B1A0D2[CE]",
  "hum_ieee_visitor": "7cab1e44-a5d9-4a77-8c69-2315820466bc",
  "hum_ieee_synced": "true",
  "s_fid": "0876A2DB3BA51B53-342C7B93A1910DE1",
  "fp": "e851336aaf80621e1915798301ce5149",
  "_cc_id": "8ba6422457dfb8a120d79665a72ebb8a",
  "__utma": "98802054.1345855402.1764297584.1764297594.1764297594.1",
  "_uetvid": "863bd130cc0311f09396559c592cfca6",
  "_ga_DRSMCND71P": "GS2.1.s1764297594$o1$g0$t1764297647$j7$l0$h0",
  "_ga_W7T9YQJQXP": "GS2.1.s1764297594$o1$g0$t1764297647$j7$l0$h0",
  "_ga_H0YHKP362D": "GS2.1.s1764297595$o1$g0$t1764297647$j8$l0$h0",
  "_ga_HNLQ4ZWTQN": "GS2.1.s1764297595$o1$g0$t1764297647$j8$l0$h0",
  "_ga_DGBMH9NVB4": "GS2.1.s1764297595$o1$g0$t1764297647$j8$l0$h0",
  "_ga_RN78LDXHRB": "GS2.1.s1764297595$o1$g0$t1764297647$j8$l0$h0",
  "cf_clearance": "Jjs9YtW3qZlxKKlEJLkKvtkQBvp8FqKlwuVR5SHlVpQ-1764919535-1.2.1.1-DtC0eYAe_lK4cUJbpbpac__8nJTRKn6wEKJW922tr6__IjkjTUx_OAoakqjVOSnfr._bJpo0qjK4Rx.AYfMqTPiQdiHgu_DPXUNixy0yXmNSx4R5ZUy2iefb4Ty70PxZtSTigBn.moWr_ZDnhK8ktYrg49kK1tG_X9HkitFQTVeqcmcSLUUL3i2s7ZdQKuHrkJ_rgD7wmqHpyOp7LSuDgFDDXksI9PjNU3d8HI_sEJg",
  "_ga": "GA1.1.1345855402.1764297584",
  "_ga_YFS85CFJD1": "GS2.1.s1764919573$o2$g1$t1764919647$j60$l0$h0",
  "utag_main": "v_id:01994c0453b3002111470324b0fa0607d002b07500bd0$_sn:25$_se:6$_ss:0$_st:1769945261163$vapi_domain:ieeexplore.ieee.org$ses_id:1769942826782%3Bexp-session$_pn:4%3Bexp-session",
  "_pubcid": "1c405103-f24c-403a-9a2d-de4558dcaaef",
  "cto_bundle": "NoUrX19DT28lMkZZWjJQV2dnWU11bVBoUGhHNFZmSXBYVmlCd05tdXgxeXc5Mndza3c4Tkp6OHEwQ2VoOUNsUFIxUUc5Sm1JNGolMkJVJTJCVHJxQ1FJaDZjcGp1NjlHNGI1dkJzY1k2V0NPS3dYZ0FQQmJTRkhzTG1tRFg0Z3paMVh0R2JMNHZjUm51dmxFeHFyY1ZrWGljSFlMZmduV0ElM0QlM0Q",
  "JSESSIONID": "13E7F53A36A18CBA4337B7EAB64A4852",
  "AWSALBAPP-1": "_remove_",
  "AWSALBAPP-2": "_remove_",
  "AWSALBAPP-3": "_remove_",
  "WLSESSION": "1862564874.47873.0000",
  "connectId": "{\"ttl\":86400000,\"lastUsed\":1784618118117,\"lastSynced\":1784618118117}",
  "kndctr_8E929CC25A1FB2B30A495C97_AdobeOrg_identity": "CiY0OTY2ODU0OTIzNjQ2MzU0MjUyMDExNDg5MDEwMzMxMzE4MDI0NlITCLTUl%2DCUMxABGAEqBFNHUDMwAPABk%2D7FnPgz",
  "kndctr_8E929CC25A1FB2B30A495C97_AdobeOrg_cluster": "sgp3",
  "aws-waf-token": "a1390621-1884-4ff0-9208-479df7d34192:BgoAhlg1GmcxAAAA:pOMAUNyC9ZToh2TlV295dhTwML05qHYU1jvJEP8iZmeWsmESvQ1yd7RDn+SgZadGVmFoeQMuxD99hkMgf533eoXL7Q+0VPzHxjMswlNxb1aBOaC9ymZ8mYmL5dRhVNmCF8Zbjv40+FUOTBYkpfEKwNazygQ9EWGwX1IPxk3UlSoTlTt8zwy1GFS3pdcPruuubc+Kej9uxuvTtxY8luvzOmI+l0xgyXKF0d8C3mFw8Xs747ZE7MQoCx/7i3psEMU/GbrUSW8/nTuG4Q7Ucw==",
  "osano_consentmanager_uuid": "20894156-6f86-49c6-a7d0-d64ab154ee76",
  "osano_consentmanager": "2PUVui_86VbpH51zEUO4AafeL_qcSxOexEKYum6pFzC0FNilzrAR_0Xo3mgFEwIlag7-QzYzOAFEKXrWsLUu1_FsjTIlk4HKaklIaztpJLymM-lpgHElA0bRa1jR5cKmitxxi1kjygPNWQ6qOupAVCPGP84mE2b4_Ua58l6jkXIxVItm3qVQ-CIJIYrRwAwGxdKh-PWX_-quXzMR9o8sv18h0hGf_zWixzOj70od3RsG7ji4JgOq6kg38wI2Aqyr9EHskEfNcCkJo4qeqtHGB0CD9crlAVVcLYv3O9y18kzXjtraEsBiuCaaUAIdvuGyXsQYD7v48sE=",
  "Adobe_ECID": "49668549236463542520114890103313180246",
  "CloudFront-Key-Pair-Id": "K3QZ9FJ2U3AXO0",
  "ERIGHTS": "x2Br2pxxVEar6mQtuQyObAB8zAC17uJERUT*4ByydUz7q75tfwJ8Ju5W1qWoTKPdKzx2BOEvqgSxxWUdSJCXJx2FUfmpzfgx3Dx3D-18x2dMDekTM6ENPt3geRSFgJRRwx3Dx3DTcx2B7NDAgOgPED4bsnhGoJwx3Dx3D-jx2BRQwhN8kR0ULj2Bg18o7wx3Dx3D-aYGmYjfNQx2BBNju7DgYWNLgx3Dx3D",
  "ipList": "\"2001:da8:d800:4aec:bcbd:870f:7ae:ae49,2001:da8:d800:91de:b16c:c3d2:af23:41bc\"",
  "__gads": "ID=b8b7cabb221ed276:T=1757917084:RT=1784621156:S=ALNI_MZUZtPw9nnvDozlFiSRNjCtWZwnDQ",
  "__gpi": "UID=00001141cbe2e7f4:T=1757917084:RT=1784621156:S=ALNI_MbZSn0X6EoO-_AlhAJSc4nZgWrfFg",
  "__eoi": "ID=f36122221f7a8660:T=1784618771:RT=1784621156:S=AA-AfjaQYsJlAMwJlujCsLbpl7dw",
  "CloudFront-Policy": "eyJTdGF0ZW1lbnQiOiBbeyJSZXNvdXJjZSI6Imh0dHBzOi8vaWVlZXhwbG9yZS5pZWVlLm9yZy9tZWRpYXN0b3JlL0lFRUUvY29udGVudC9tZWRpYS8xMTMyMjgwOC8xMTMyMzI1My8xMTMyMzQ2NS8qIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzg0NjIzMDY1fSwiSXBBZGRyZXNzIjp7IkFXUzpTb3VyY2VJcCI6IjIwMDE6ZGE4OmQ4MDA6OTFkZTpiMTZjOmMzZDI6YWYyMzo0MWJjIn19fV19",
  "CloudFront-Signature": "fjiwRHDwL5inB7LldnnWS3j63066YJpkK-rTg-IJA49gUsCS1wSUE6ctgoBhxZUTg~KpTD9N~XMVlyEepjYfarEoM0RAP7ABZ1KhBwMYn6maH2dGGhNIEeH5yyY2bn22De3rBfPrze0A-1I4G9R2h02E5gNMke0F5IGE9i~l83j4zopn4NtABzakO~5x~htALjNpLyZoZpgn83JTFR5oTUAN2zC550GOFqoiUpGbK04HpEgxEEWxwyzkdQN-79CMgcXI01mdC3Fh-5dVzHzyCG5UL~gxfH58keIXwk5IvlCmIK~XDvpmAYV9bSIa-f8dpNgAWxnv6IWDG~y8RV3WHQ__",
  "xpluserinfo": "eyJpc0luc3QiOiJ0cnVlIiwiaW5zdE5hbWUiOiJVbml2ZXJzaXR5IG9mIFNjaWVuY2UgJiBUZWNobm9sb2d5IG9mIENoaW5hIiwicHJvZHVjdHMiOiJJQk06MTg3MjoyMDIwfElFTHxWREV8Tk9LSUEgQkVMTCBMQUJTfCJ9",
  "seqId": "8875",
  "AWSALBAPP-0": "AAAAAAAAAABLSpDZWO4NGpH/wjSVjTHEvr+YD2Ya8msOXSpWo2zGvwoF/PFwuV4/yKQ37n2llfWk58IUj03DBCtLSnNrXgt4PKW+nol/zGFiSvcNH5Hf3+QLq2xOc+PZdv4tBYOmMRSh7UXibCgjmKEAtCslkooBEZLwo2PEkoCgDEnCywgpsNd/xi+EeLyQVef2CnRuzrLA+t+kq+5JFQ==",
  "TS016349ac": "01f15fc87c87c4ecd0b70a157ad0d183b09e1e20db84bbc097090e805dda3ee488faaf6fc19036cbd4fbff5f2a31a5830fd52e0bf3",
  "TS8b476361027": "0807dc117eab2000bdcf094a77f0846ed7a42c6a956112e1287b774f227a98462e04545c960eb5c408adadfd0b1130006fc7b8c9b7b9360543f363be5d067efeb9ee2ed0d8c75c79c2b54c6f1c4b76ce36d317597b00b197d42a09aab3d987b1"
}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for title, pdf_url in tqdm([(title,pdf_url)], desc="下载PDF（代理版）"):
        try:
            # 使用代理发送请求
            response = requests.get(
                pdf_url,
                stream=True,
                # proxies=proxies,
                headers=headers,
                # cookies=cookies,
                timeout=30  # 设置超时时间
            )

            if response.status_code == 200:
                # 替换文件名中的非法字符
                safe_title = "".join(c if c.isalnum() else "_" for c in title)
                file_path = os.path.join(save_dir, f"{safe_title}.pdf")

                # 分块写入文件
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)

                doc = fitz.open(file_path)
                full_text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                doc.close()
                return full_text

            else:
                print(f"下载失败: {title} | 状态码: {response.status_code} | URL: {pdf_url}")
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {title} | 错误: {e}")
        except Exception as e:
            print(f"未知错误: {title} | 错误: {e}")




    # # 只留 organic 里有用的字段（跟上一轮你贴的结构对齐）
    # cleaned = []
    # for item in raw.get("organic", []):
    #     cleaned.append({
    #         "title": item.get("title"),
    #         "link": item.get("link"),
    #         "publicationInfo": item.get("publicationInfo"),
    #         "snippet": item.get("snippet"),
    #         "year": item.get("year"),
    #         "citedBy": item.get("citedBy"),
    #         "pdfUrl": item.get("pdfUrl"),
    #     })
    #
    # return json.dumps(cleaned, ensure_ascii=False)


# ── Tool Schema（DeepSeek 兼容 OpenAI tools 协议）───────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_google_scholar",
            "description": "用 Google Scholar 搜索中文学术 / 英文论文，返回标题、作者、年份、被引量等。当用户问到论文、研究、文献、某作者工作时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "学术搜索关键词，如 'Apple Inc international marketing 2024'",
                    },
                    "num": {
                        "type": "integer",
                        "description": "返回条数，默认 10，最大 20",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
{
        "type": "function",
        "function": {
            "name": "read_paper",
            "description": "根据论文的PDF链接来下载论文",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_url": {
                        "type": "string",
                        "description": "论文pdf的url",
                    },
                    "title": {
                        "type": "string",
                        "description": "论文名称，即要保存的pdf文件名称",
                    },
                },
                "required": ["pdf_url", "title"],
            },
        },
    },
]


# ── Agent Loop ──────────────────────────────────────────
def agent_run(user_query: str, model: str = "deepseek-v4-pro"):
    """
    deepseek-chat = V3.2 非思考模式
    deepseek-reasoner = V3.2 思考模式（tool call 时要回传 reasoning_content，见下方提示）
    """
    messages = [
        {"role": "system", "content": "你是一个学术辅助助手，擅长用 Google Scholar 检索论文并下载以及做综述。"},
        {"role": "user", "content": user_query},
    ]

    # 第一轮：让模型决定是否调工具
    resp = deepseek.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    while True:
        # print("========================思考过程========================\n")
        # print(resp.get('reasoning_content', ''))
        # print("====================================================\n")

        tool_calls = msg.tool_calls
        messages.append(msg)

        # 如果调了 tool，执行后回传
        if tool_calls:
            print("调用了工具：", tool_calls)

            for tc in tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = ""
                if fn_name == "search_google_scholar":
                    result = search_google_scholar(**args)
                elif fn_name == "read_paper":
                    result = read_paper(**args)
                else:
                    result = json.dumps({"error": "unknown tool"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

            # 第二轮：模型拿到 Scholar 结果后做自然语言回答
            resp = deepseek.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            # return final.choices[0].message.content
        # 如果未调用，则跳出循环
        else:
            # responses.append(resp.get('content', ''))
            # think_responses.append(resp.get('reasoning_content', ''))
            return msg.content


    # # 如果模型发了 tool_calls
    # if msg.tool_calls:
    #     for tc in msg.tool_calls:
    #         fn_name = tc.function.name
    #         args = json.loads(tc.function.arguments)
    #
    #         if fn_name == "read_paper":
    #             result = read_paper(**args)
    #         else:
    #             result = json.dumps({"error": "unknown tool"})
    #
    #         # 把『模型发的调用』和『工具返回』都追加进上下文
    #         messages.append(msg)          # assistant 那条（含 tool_calls）
    #         messages.append({
    #             "role": "tool",
    #             "tool_call_id": tc.id,
    #             "content": result,
    #         })
    #
    #     # 第二轮：模型拿到 Scholar 结果后做自然语言回答
    #     final = deepseek.chat.completions.create(
    #         model=model,
    #         messages=messages,
    #     )
    #     return final.choices[0].message.content

    # 没触发工具，直接答
    return msg.content


# ── 试运行 ──────────────────────────────────────────────
if __name__ == "__main__":
    # q = "Apple Inc 营销战略 学术论文"
    # answer = agent_run(q)
    # print("\n🧠 DeepSeek 回答：\n")
    # print(answer)


    pdf_links="https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=11323465"
    read_paper([("LLMSR", pdf_links)])