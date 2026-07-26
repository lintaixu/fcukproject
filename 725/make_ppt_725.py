# -*- coding: utf-8 -*-
"""產生 725/進度報告.pptx — 白底黑字極簡風; 主軸: 提高檔數 + 分族群。"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

BLACK = RGBColor(0, 0, 0)
GRAY = RGBColor(0x52, 0x51, 0x4E)
FONT = "微軟正黑體"
HERE = os.path.dirname(os.path.abspath(__file__))

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide(title, lines=None, title_size=32):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.1), Inches(1.0))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = title
    r.font.name = FONT; r.font.size = Pt(title_size); r.font.bold = True
    r.font.color.rgb = BLACK
    if lines:
        body = s.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.8), Inches(5.6))
        tf = body.text_frame
        tf.word_wrap = True
        for i, (text, size, indent) in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.level = indent
            p.space_after = Pt(10)
            r = p.add_run(); r.text = text
            r.font.name = FONT; r.font.size = Pt(size); r.font.color.rgb = BLACK
    return s


def add_table(slide, rows, top=1.6, left=0.8, width=11.7, col_widths=None, font_size=15):
    n_r, n_c = len(rows), len(rows[0])
    shp = slide.shapes.add_table(n_r, n_c, Inches(left), Inches(top), Inches(width),
                                 Inches(0.42 * n_r))
    tbl = shp.table
    if col_widths:
        for j, w in enumerate(col_widths):
            tbl.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i else RGBColor(0xEF, 0xEE, 0xEA)
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(val)
            r.font.name = FONT; r.font.size = Pt(font_size)
            r.font.bold = (i == 0)
            r.font.color.rgb = BLACK
    return tbl


# 1. 標題
s = add_slide("本週進度報告:提高檔數與分族群實驗", title_size=40)
tb = s.shapes.add_textbox(Inches(0.6), Inches(3.2), Inches(12), Inches(1.5))
for txt, size in [("2026 / 07 / 27", 24),
                  ("核心問題:加更多股票、或改用同族群股票,能否突破 ~50%?", 18)]:
    p = tb.text_frame.add_paragraph()
    r = p.add_run(); r.text = txt
    r.font.name = FONT; r.font.size = Pt(size); r.font.color.rgb = GRAY

# 2. 本週主軸
add_slide("本週做了什麼", [
    ("主軸一:提高檔數 — 建立 TW500 池(554 檔上市普通股), 50→500 每次 +50 共 10 輪", 21, 0),
    ("主軸二:分族群 — 盤點電子族群 455 檔;全電子 / 半導體 / 電子零組件 / 金融對照 四組實驗", 21, 0),
    ("前置工作:發現並修正評估協定的未來洩漏(本週所有數字均為乾淨協定)", 21, 0),
    ("其他:XGB 同條件對照、論文網格重掃、論文原文核對、工程與文件整理", 21, 0),
    ("交付:報告與操作文檔(725/)、實驗記錄與全部 JSON、規模曲線圖", 21, 0),
])

# 3. 前置:協定修正
s = add_slide("前置:評估協定修正(為什麼數字可信)")
body = s.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.8), Inches(1.7))
tf = body.text_frame; tf.word_wrap = True
for txt in [
    "論文的 self-attention 對「同一批樣本」互相運算,原本連續切批會讓「明天的樣本」與「今天」同批 — 間接看到答案。",
    "改為 date 批次(同日所有股票一批):無洩漏,且與論文「探索股票間關聯」文字相容。",
]:
    p = tf.add_paragraph()
    r = p.add_run(); r.text = "•  " + txt
    r.font.name = FONT; r.font.size = Pt(19); r.font.color.rgb = BLACK
add_table(s, [
    ["評估方式(同一顆模型)", "同批有無未來樣本", "Test Acc"],
    ["序向 128 筆(舊協定)", "有", "52.70%"],
    ["同日一批(新標準)", "無", "51.30%"],
    ["一次一筆", "完全無跨樣本", "46.82%"],
], top=3.4, col_widths=[5.5, 3.4, 2.8])

# 4. 規模實驗(圖)
s = add_slide("主軸一:提高檔數 50 → 500(10 輪)")
img = os.path.join(HERE, "scale_tw500_curve.png")
if os.path.exists(img):
    s.shapes.add_picture(img, Inches(1.35), Inches(1.3), width=Inches(10.6))
tb = s.shapes.add_textbox(Inches(0.8), Inches(6.8), Inches(11.8), Inches(0.5))
p = tb.text_frame.paragraphs[0]
r = p.add_run()
r.text = "TW500 巢狀池(前 50 = TW50);訓練樣本 9 萬 → 88.5 萬(10 倍)"
r.font.name = FONT; r.font.size = Pt(15); r.font.color.rgb = GRAY

# 5. 規模實驗結論
add_slide("規模實驗結論:資料量不是瓶頸", [
    ("樣本增加 10 倍,Acc 46~56 徘徊、F1 macro 35.9~47.4 兩態震盪 — 無任何規模趨勢", 22, 0),
    ("Acc 的「上升」是假象:大池子「跌」類比例較高,全押跌的地板就更高", 22, 0),
    ("n=500 時模型完全塌縮:一張「漲」的預測都不出(F1 macro 35.9% = 數學下限)", 22, 0),
    ("回測 10 輪有 9 輪輸給等權買入持有", 22, 0),
])

# 6. 族群實驗
s = add_slide("主軸二:分族群(電子 455 檔盤點 + 四組實驗)")
body = s.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.8), Inches(1.0))
p = body.text_frame.paragraphs[0]
r = p.add_run()
r.text = ("•  盤點:上市電子相關 455 檔(零組件 104、半導體 96、光電 68、電腦週邊 64、其他 46、"
          "通信 45、通路 21、資服 11)")
r.font.name = FONT; r.font.size = Pt(18); r.font.color.rgb = BLACK
add_table(s, [
    ["族群", "檔數", "Acc", "F1 macro", "回測超額", "形態"],
    ["全電子", "450", "51.29%", "51.22%", "+0.4%", "均衡 ≈ 亂猜(歷來最高 F1m)"],
    ["半導體", "92", "44.85%", "38.24%", "-11.6%", "反向塌縮(全押漲)"],
    ["電子零組件", "104", "49.33%", "48.91%", "-3.8%", "均衡亂猜"],
    ["金融保險(對照)", "32", "54.16%", "39.09%", "-18.4%", "塌縮(全押跌)"],
], top=2.5, col_widths=[2.6, 1.2, 1.6, 1.8, 1.7, 2.8], font_size=16)
body = s.shapes.add_textbox(Inches(0.8), Inches(5.3), Inches(11.8), Inches(1.2))
p = body.text_frame.paragraphs[0]
r = p.add_run()
r.text = "•  沒有任何一組同時做到「Acc 高於類別先驗 + 預測均衡」;同質性最高的半導體反而最差 — 分族群不改變天花板"
r.font.name = FONT; r.font.size = Pt(19); r.font.color.rgb = BLACK

# 7. 結論
add_slide("總結論", [
    ("否證鏈四維完整:", 24, 0),
    ("換市場(上證 50)✗   掃參數(論文全網格)✗   加資料量(10 倍)✗   分族群 ✗", 21, 1),
    ("無洩漏協定下,Chart-GCN 於 1 日漲跌標籤與 XGBoost、擲硬幣無法區分", 21, 0),
    ("瓶頸 = 1 日 close-to-close 標籤的資訊含量,不是模型 / 參數 / 資料量 / 股票組成", 21, 0),
])

# 8. 下一步
add_slide("下一步(待討論)", [
    ("1. 換更可預測的題目:5 日標籤(降噪)、橫斷面相對強弱標籤", 21, 0),
    ("2. 回到融合主軸:main(AE+OCSVM)× Chart-GCN", 21, 0),
    ("3. 引入新資訊源:籌碼 / 分點特徵 — 價格衍生特徵的資訊已證明不足", 21, 0),
    ("", 14, 0),
    ("細節:725/報告_本週工作.md、725/操作文檔.md、實驗記錄_論文對齊.md", 16, 0),
])

out = os.path.join(HERE, "進度報告.pptx")
prs.save(out)
print("saved", out)
