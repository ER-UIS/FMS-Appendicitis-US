# -*- coding: utf-8 -*-
"""
报告原文特征预提取程序（report_text_extract.py）
================================================
只依据超声报告"描述"原文，按《报告原文特征提取标注规则》预提取九项特征，供人工逐条核对。
不读取图像、病理、化验、报告标签、FMO 或专家结果（盲法）。

每项特征输出四种状态之一：
  阳性      报告文字明确描述该异常
  阴性      报告文字明确否定该异常（如"未见粪石""腹膜腔内未探及游离液性暗区""阑尾外径 4 mm"）
  未提及    报告未涉及该特征
  待核对    程序无法可靠判断（阴阳性语句并存、含"可疑/考虑/不除外"、外径 6.0–6.9 mm 需按年龄判断、
            右下腹包块而未写"阑尾"等），必须人工判定
最终二分类：阳性 = 阳性；阴性 = 阴性或未提及（常规报告未写即视为未报告该异常）。

用法：python report_text_extract.py  <输入xlsx(含 申请号, 描述 两列)>  <输出xlsx>
"""
import re
import sys

FEATURES = ['阑尾外径', '粪石', '腹腔积液', '阑尾壁情况', '阑尾腔内积气', '阑尾腔内积液', '阑尾周边积液', '阑尾周围系膜', '阑尾区血流']
CODES = ['AODC', 'CC', 'CAE', 'AWC', 'GAC', 'FAC', 'PFCC', 'MSC', 'BFC']

NEG = r'(未见|未探及|未显示|未发现|无明显|无|没有|不伴|未)'
HEDGE = r'(可疑|疑似|考虑|不除外|不排除|似|可能|待排)'
APPX_LINE = r'阑尾|右下腹|麦氏|回盲|同心圆'


def clean(s):
    s = str(s or '').replace('_x000D_', '').replace('\r', '\n')
    s = s.replace('：', ':').replace('；', ';').replace('，', ',').replace('。', '.')
    s = re.sub(r'[ \t　]+', '', s)
    return s


def lines(s):
    return [l for l in re.split(r'\n+', s) if l.strip()]


def sentences(text):
    return [x for x in re.split(r'[.;]', text) if x]


def clauses(sent):
    return [x for x in re.split(r',', sent) if x]


def appendix_text(s):
    """阑尾相关段落：含 阑尾/右下腹/麦氏/回盲/同心圆 的行；去掉描述淋巴结的句子。"""
    keep = [l for l in lines(s) if re.search(APPX_LINE, l)]
    txt = '\n'.join(keep)
    sents = [x for x in sentences(txt) if '淋巴结' not in x]
    return '.'.join(sents)


def negated(clause, start, end=None):
    """关键词前 8 个字内或匹配片段内出现否定词。"""
    seg = clause[max(0, start - 8):(end if end is not None else start)]
    return re.search(NEG, seg) is not None


def scan(text, pos_pats, neg_pats=(), exclude=None, clause_exclude=None):
    """逐分句扫描，返回 (阳性分句列表, 阴性分句列表, 含模糊词的阳性分句列表)。"""
    P, N, H = [], [], []
    for sent in sentences(text):
        if exclude and re.search(exclude, sent):
            continue
        for c in clauses(sent):
            if clause_exclude and re.search(clause_exclude, c):
                continue
            hit_neg = any(re.search(p, c) for p in neg_pats)
            m = None
            for p in pos_pats:
                m = re.search(p, c)
                if m:
                    break
            if m and not negated(c, m.start(), m.end()) and not hit_neg:
                (H if re.search(HEDGE, c) else P).append(c)
            elif hit_neg or (m and negated(c, m.start(), m.end())):
                N.append(c)
    return P, N, H


def decide(P, N, H, extra_flag=None, pos_wins=None):
    ev = ' | '.join((P + H + N)[:3])
    if extra_flag:
        return '待核对', f'{extra_flag}; {ev}'.strip('; ')
    if H and not P:
        return '待核对', '含可疑/考虑等措辞: ' + ev
    if P and N and pos_wins:
        return '阳性', f'{pos_wins}: ' + ev
    if P and N:
        return '待核对', '阳性与否定语句并存: ' + ev
    if P:
        return '阳性', ev
    if N:
        return '阴性', ev
    return '未提及', ''


def f_aodc(a, full):
    P, N, H = scan(a, [r'(肿胀|增粗|肿大|增大|增宽)的?阑尾', r'阑尾.{0,6}(肿胀|增粗|肿大|增大)', r'远段肿胀'],
                   [r'阑尾.{0,10}未见(明显)?(肿胀|增粗)', r'阑尾(区)?未见(明显)?异常(?!.{0,2}(彩流|血流))', r'^(阑尾)?未见(明显)?(肿胀|增粗)'],
                   clause_exclude=r'CDFI|血流|彩流|回盲部|肠壁')
    diams = [float(x) for x in re.findall(r'外径约?(\d+(?:\.\d+)?)mm', a)]
    flag = None
    if diams:
        d = max(diams)
        if d >= 7:
            P = P or [f'外径 {d:g} mm']
        elif d >= 6:
            if not P and not N:
                flag = f'外径 {d:g} mm，未写是否肿胀，需按年龄判断（<18 岁 ≥6 mm 为阳性）'
        else:
            if not P:
                N = N or [f'外径 {d:g} mm']
    if not P and not N and not diams:
        m = re.search(r'(右下腹|阑尾区).{0,20}?(\d+(?:\.\d+)?)\*(\d+(?:\.\d+)?)mm.{0,12}?(包块|低回声|低无回声|混合回声|块影|回声区)', a) or \
            re.search(r'(右下腹|阑尾区).{0,20}?(包块|低回声|低无回声|混合回声|块影).{0,20}?(\d+(?:\.\d+)?)\*(\d+(?:\.\d+)?)mm', a)
        if m:
            flag = '阑尾区/右下腹包块或低回声区（只给出病灶大小，未写外径或增粗）'
    st, ev = decide(P, N, H, flag)
    return st, ev


def f_cc(a, full):
    P, N, H = scan(a, [r'粪石', r'强(回声|光)(团|光团|斑).{0,8}声影', r'强(回声|光)(团|光团).{0,4}后方?伴'],
                   [r'未见(明显)?粪石', r'腔内未见(明显)?强(回声|光)'])
    flag = None
    if not P and not N and re.search(r'(低回声区|包块|阑尾).{0,20}(强光团|强回声团|强回声)', a):
        flag = '阑尾/右下腹病灶内强回声，未写声影或粪石'
    return decide(P, N, H, flag)


def f_cae(a, full):
    P, N, H = scan(full, [r'(腹腔|腹膜腔|盆腔|肠间隙|肠间|肝肾间隙|脾肾间隙|子宫直肠|陶氏|道格拉斯|膀胱后方|膀胱周围|肝周|脾周).{0,12}(可见|可探及|探及|见|有).{0,12}(液性暗区|液暗区|积液|游离液)',
                             r'腹腔积液', r'盆腔积液', r'腹水'],
                      [r'(腹腔|腹膜腔|盆腔).{0,4}(内)?未(探及|见)(明显)?(游离)?(液性暗区|液暗区|积液)', r'未见(明显)?腹腔积液'],
                      exclude=r'胆囊|肾盂|囊肿|卵巢内|子宫内|附件区.{0,6}囊|关节',
                      clause_exclude=r'阑尾|周边|周围|旁')
    return decide(P, N, H, pos_wins='局部游离积液优先于总体否定句')


def f_awc(a, full):
    P, N, H = scan(a, [r'壁.{0,8}(连续性中断|中断|不连续|连续性欠佳|分层(结构)?(消失|不清|模糊)|层次(不清|消失|模糊)|毛糙|模糊|缺损|破溃|欠完整|不完整)',
                       r'穿孔', r'壁.{0,4}回声(不均|减低)'],
                   [r'壁.{0,4}(连续|光滑|层次清晰|分层清晰|完整)(?!性)'],
                   clause_exclude=r'肠壁|回盲|胆囊|膀胱|子宫|胃壁')
    return decide(P, N, H)


def f_gac(a, full):
    P, N, H = scan(a, [r'(腔内|管腔|阑尾内|壁内|阑尾).{0,10}(积气|气体|气体样?强回声|气体回声)', r'积气'],
                   [r'(腔内)?未见(明显)?积气'], exclude=r'肠气|胃肠气|气体干扰|肠腔积气|消化道|肠道|胃')
    return decide(P, N, H)


def f_fac(a, full):
    P, N, H = scan(a, [r'腔内.{0,4}积液', r'腔内.{0,6}液(性)?暗区', r'管腔.{0,6}(扩张|增宽).{0,8}(积液|液)', r'(腔内|内)透声(差|欠佳|不佳)',
                       r'腔内.{0,6}(积脓|脓液)'],
                   [r'腔内未见(明显)?(积液|液)', r'管腔萎瘪'],
                   clause_exclude=r'腹膜腔|腹腔|盆腔|宫腔|膀胱|胆囊|肠管|肠腔')
    return decide(P, N, H)


def f_pfcc(a, full):
    P, N, H = scan(a, [r'(周边|周围|阑尾周|阑尾旁).{0,10}(积液|液(性)?暗区|渗出|液体)'],
                   [r'(周边|周围).{0,4}未(见|探及)(明显)?(积液|液(性)?暗区|渗出)'])
    return decide(P, N, H)


def f_msc(a, full):
    P, N, H = scan(a, [r'系膜.{0,10}(肿胀|增厚|回声增强|增强)', r'网膜.{0,4}(包裹|聚集|增厚|肿胀)', r'(周围|周边)脂肪.{0,6}(回声增强|增强|增厚)'],
                   [r'系膜.{0,4}未见(明显)?(肿胀|增厚)'], exclude=r'系膜淋巴结')
    return decide(P, N, H)


def f_bfc(a, full):
    P, N, H = scan(a, [r'(阑尾|壁|其内|周边|周围|包块|低回声区|病灶|管壁).{0,12}(可见|探及|见|测及).{0,10}(血流|彩流)',
                       r'血流(信号)?(丰富|增多|增加|较丰富)', r'彩流(信号)?(丰富|增多)'],
                   [r'未(见|探及)(明显)?(异常)?(血流|彩流)', r'无(明显)?(血流|彩流)', r'(血流|彩流)(信号)?未见'],
                   clause_exclude=r'肾|肝|脾|胆|膀胱|前列腺|子宫|卵巢|淋巴结')
    st, ev = decide(P, N, H, pos_wins='阑尾/病灶处探及血流优先')
    if st == '阴性' and re.search(r'腹腔内未见(明显)?异常(彩流|血流)', ev):
        ev = '（模板句）' + ev
    return st, ev


FUNCS = [f_aodc, f_cc, f_cae, f_awc, f_gac, f_fac, f_pfcc, f_msc, f_bfc]


def extract(report):
    full = clean(report)
    a = appendix_text(full)
    out = {'阑尾相关段落': a.replace('.', '。').replace(',', '，')}
    if not a:
        out['段落提示'] = '报告中未找到阑尾/右下腹段落'
    ds = [float(x) for x in re.findall(r'外径约?(\d+(?:\.\d+)?)mm', a)]
    out['外径测量值_程序'] = max(ds) if ds else None
    for f, fn in zip(FEATURES, FUNCS):
        st, ev = fn(a, full)
        out[f + '_程序'] = st
        out[f + '_依据'] = ev.replace('.', '。').replace(',', '，')
    return out


if __name__ == '__main__':
    import pandas as pd
    src, dst = sys.argv[1], sys.argv[2]
    df = pd.read_excel(src)
    rows = []
    for pid, rep in zip(df['申请号'].astype(str), df['描述']):
        r = {'申请号': pid}; r.update(extract(rep)); rows.append(r)
    pd.DataFrame(rows).to_excel(dst, index=False)
    print('saved', dst, len(rows))
