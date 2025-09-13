'use client';

import { useEffect, useMemo, useState } from 'react';
import Papa from 'papaparse';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, LabelList
} from 'recharts';
import {
  format, parseISO, parse as parseWithFormat, isValid,
  subMonths, subDays, startOfDay, endOfDay
} from 'date-fns';

type RawRow = Record<string, string>;
type Row = {
  date: string;
  product_name: string;
  product_category?: string;
  quantity: number;
  revenue: number;
  customer_name?: string;
  customer_email?: string;
  order_id?: string;

  unit_price?: number | null;
  list_price?: number | null;
  discount_amount?: number | null;
  coupon_code?: string | null;
  shipping_city?: string | null;
  was_discounted?: boolean;
};
type Monthly = { yyyymm: string; revenue: number; orders: number };
type TopItem = { product_name: string; revenue: number };
type TopCat = { category: string; revenue: number };

type AssocRule = { antecedents: string[]; consequents: string[]; support: number; confidence: number; lift: number };
type ServerInsights = {
  assoc_rules: AssocRule[];
  rfm_share: { low: number; mid: number; high: number };
  rfm_counts?: { low: number; mid: number; high: number };
  rfm_avg_monetary?: { low: number; mid: number; high: number };
  potential_products: { product_name: string; mom_last: number; sum3_revenue: number }[];
  nurture_list: { customer_email: string; last_order_date: string; lifetime_orders: number; lifetime_revenue: number }[];
  nurture_mid?: { customer_email: string; last_order_date: string; lifetime_orders: number; lifetime_revenue: number }[];
  nurture_low?: { customer_email: string; last_order_date: string; lifetime_orders: number; lifetime_revenue: number }[];
  diagnostics?: {
    assoc: { ok: boolean; baskets_total: number; baskets_valid: number; reason: string | null };
    rfm: { ok: boolean; customers_total: number; customers_used: number; reason: string | null };
    required?: { ok: boolean; reason?: string; missing?: string[] };
  };
};

/* === 欄位 & 數值處理 === */
export const REQUIRED = ['date','product_name','revenue'] as const;

export const ALIASES: Record<string, string[]> = {
  date: ['date','order date','invoice date','dt','order_date','invoice_date','日期','訂單日期','交易日期','銷售日期'],
  product_name: ['product_name','product name','product','item','product line','sku name','sku','item name','lineitem name','商品名稱','產品名稱','品名','商品','項目','品項'],
  revenue: ['revenue','sales','total','amount','sales amount','total price','gross sales','gross income','line total','extended price','subtotal','營收','金額','成交金額','銷售金額','實收','應收','訂單金額','總額','付款金額'],

  product_category: ['product_category','category','product line','lineitem_category','cat','dept','department','商品類別','類別','分類','部門'],
  quantity: ['quantity','qty','units','count','quantity ordered','order qty','order quantity','lineitem quantity','數量','件數','購買數','購買數量'],

  unit_price: ['lineitem price','line item price','unit price','price','sale price','單價','成交單價'],
  list_price: ['compare at price','list price','original price','msrp','原價','標價'],
  discount_amount: ['discount','discount amount','lineitem discount','promotion discount','promo discount','折扣','折扣金額'],
  coupon_code: ['coupon','coupon code','promo code','promotion code','優惠碼','折扣碼'],

  customer_name: ['customer_name','name','customer','customer name','buyer','client name','客戶姓名','顧客姓名','買家'],
  customer_email: ['customer_email','email','e-mail','customer email','buyer email','電子郵件','email信箱','信箱'],
  order_id: ['order_id','order number','invoice id','order id','invoice','order_no','order-no','訂單編號','訂單號'],

  shipping_city: ['shipping city','ship city','city','shipping address city','recipient city','配送城市','收件城市','城市']
};

function pick(row: Record<string, string>, canonical: keyof typeof ALIASES) {
  const normMap = Object.fromEntries(
    Object.entries(row).map(([k, v]) => [k.replace(/^\uFEFF/, '').toLowerCase().trim(), v])
  );
  for (const alias of ALIASES[canonical]) {
    const key = alias.toLowerCase().trim();
    if (normMap[key] !== undefined) return normMap[key];
  }
  return '';
}

const DATE_FORMATS = [
  'yyyy-MM-dd', 'yyyy/M/d', 'yyyy/MM/dd',
  'M/d/yyyy','d/M/yyyy','MM/dd/yyyy','dd/MM/yyyy',
  'M/d/yy','d/M/yy'
];

function toISODate(s: string): string | null {
  if (!s) return null;
  let t = String(s).trim().replace('T', ' ');
  if (t.includes(' ')) t = t.split(' ')[0];
  for (const f of DATE_FORMATS) {
    const d = parseWithFormat(t, f, new Date());
    if (isValid(d)) return format(d, 'yyyy-MM-dd');
  }
  const d2 = new Date(t);
  return isValid(d2) ? format(d2, 'yyyy-MM-dd') : null;
}

function toNumber(x: any) {
  if (x === null || x === undefined) return 0;
  let s = String(x).trim();
  if (!s) return 0;
  const neg = /^\(.*\)$/.test(s);
  s = s.replace(/[\(\)]/g, '');
  s = s.replace(/[^\d.\-]/g, '');
  if (!s || s === '.' || s === '-') return 0;
  const n = Number(s);
  return Number.isFinite(n) ? (neg ? -n : n) : 0;
}

function normalizeRow(r: RawRow): Row | null {
  const iso = toISODate(pick(r, 'date'));
  const normMap = Object.fromEntries(Object.entries(r).map(([k, v]) => [k.replace(/^\uFEFF/, '').toLowerCase().trim(), v]));
  const product_name = String(pick(r, 'product_name') || normMap['lineitem name'] || '').trim();

  const product_category = String(pick(r, 'product_category') || '').trim();
  const quantity = (() => {
    const q = toNumber(pick(r, 'quantity'));
    if (Number.isFinite(q) && q > 0) return q;
    return 1;
  })();

  let revenue = toNumber(pick(r, 'revenue'));
  const unit_price_from_alias = toNumber(pick(r, 'unit_price'));
  const list_price_from_alias = toNumber(pick(r, 'list_price'));
  const discount_amount = toNumber(pick(r, 'discount_amount'));
  const coupon_code = (pick(r, 'coupon_code') || '').trim() || null;

  if ((revenue === 0 || !Number.isFinite(revenue)) && Number.isFinite(unit_price_from_alias)) {
    revenue = unit_price_from_alias * (quantity || 1);
  }

  const unit_price = Number.isFinite(unit_price_from_alias) && unit_price_from_alias > 0
    ? unit_price_from_alias
    : (quantity > 0 ? revenue / quantity : undefined);

  const list_price = Number.isFinite(list_price_from_alias) && list_price_from_alias > 0
    ? list_price_from_alias
    : undefined;

  const was_discounted =
    (!!discount_amount && discount_amount > 0) ||
    (!!list_price && !!unit_price && list_price > unit_price * 1.01) ||
    (!!coupon_code && coupon_code.length > 0) || false;

  const customer_name = String(pick(r, 'customer_name') || '').trim();
  const customer_email = String(pick(r, 'customer_email') || '').trim();
  const order_id = String(pick(r, 'order_id') || '').trim();

  const shipping_city_raw = String(pick(r, 'shipping_city') || '').trim();
  const shipping_city = shipping_city_raw ? shipping_city_raw : null;

  if (!iso || !product_name) return null;
  return {
    date: iso,
    product_name,
    product_category,
    quantity,
    revenue,
    customer_name,
    customer_email,
    order_id,
    unit_price: unit_price ?? null,
    list_price: list_price ?? null,
    discount_amount: discount_amount || null,
    coupon_code,
    shipping_city,
    was_discounted,
  };
}

/* ===== 主頁 ===== */
type Tab = 'summary' | 'customers' | 'products';

export default function Home() {
  useEffect(() => {
    document.documentElement.classList.remove('dark');
    document.body.classList.add('bg-white', 'overflow-x-hidden');
  }, []);

  const [rows, setRows] = useState<Row[]>([]);
  const [fileName, setFileName] = useState<string>('');
  const [notice, setNotice]   = useState<{level:'error'|'warn'|'ok'; msg:string} | null>(null);
  const [server, setServer]   = useState<ServerInsights | null>(null);
  const [assocD11, setAssocD11] = useState<AssocRule[] | null>(null);
  const [assoc618, setAssoc618] = useState<AssocRule[] | null>(null);
  const [serverNote, setServerNote] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('summary');
  const [deckBusy, setDeckBusy] = useState(false);
  const [deckMsg, setDeckMsg] = useState<string | null>(null);

  const onFile = async (f: File) => {
    setFileName(f.name); setServer(null); setServerNote(null); setNotice(null);

    const buf = await f.arrayBuffer();

    let text = new TextDecoder('utf-8', { fatal: false }).decode(buf);
    const bad = (text.match(/\uFFFD/g) || []).length;
    if (bad > text.length * 0.01) {
      try {
        text = new TextDecoder('big5').decode(buf);
      } catch {
      }
    }

    Papa.parse<RawRow>(text, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (h: string) => (h ?? '').toString().replace(/^\uFEFF/, '').toLowerCase().trim(),
      complete: async (res) => {
        const headers = (res.meta.fields || []).map(h => (h||'').toString());
        const missing: string[] = [];
        for (const k of REQUIRED) {
          const hit = headers.some(h => ALIASES[k].includes(h.toLowerCase().trim()));
          if (!hit) missing.push(k);
        }
        if (missing.length) {
          setRows([]); setNotice({level:'error', msg:`缺少必填欄位：${missing.join(', ')}`}); return;
        }

        let badDate=0, badName=0, badNum=0;
        const cleaned = (res.data as RawRow[]).map(r=>{
          const iso = toISODate(pick(r,'date'));
          const name = String(pick(r,'product_name')||'').trim();
          const qty = toNumber(pick(r,'quantity'));
          const rev = toNumber(pick(r,'revenue'));
          if (!iso) badDate++; if (!name) badName++;
          if (!Number.isFinite(qty) || !Number.isFinite(rev)) badNum++;
          return normalizeRow(r);
        }).filter((x): x is Row => !!x);

        if (cleaned.length === 0) {
          setNotice({level:'error', msg:`必要欄位格式無效：日期不可解析 ${badDate} 列、產品名稱空白 ${badName} 列、數值錯誤 ${badNum} 列`});
          return;
        }
        if (badDate || badName || badNum) setNotice({level:'warn', msg:`已略過不合法列：日期 ${badDate}、名稱 ${badName}、數值 ${badNum}`});
        else setNotice({level:'ok', msg:`已載入 ${cleaned.length} 列資料`});

        setRows(cleaned);
        analyzeOnServer(cleaned);
      },
      error: (err) => setNotice({level:'error', msg:`CSV 解析失敗：${err.message}`}),
    });
  };

    const monthlyAll: Monthly[] = useMemo(() => {
    const map = new Map<string, { revenue: number; orders: number }>();
    for (const r of rows) {
      const yyyymm = format(parseISO(r.date), 'yyyy-MM');
      const cur = map.get(yyyymm) || { revenue: 0, orders: 0 };
      map.set(yyyymm, { revenue: cur.revenue + r.revenue, orders: cur.orders + 1 });
    }
    return [...map.entries()].sort(([a],[b])=>a.localeCompare(b)).map(([yyyymm,v])=>({ yyyymm, ...v }));
  }, [rows]);

  useEffect(() => {
    if (!rows.length) { setAssocD11(null); setAssoc618(null); return; }

    const y = getLastYearKeyDate();
    const wD11 = double11Window(y);
    const w618 = six18Window(y);

    (async () => {
      try {
        const [r1, r2] = await Promise.all([
          fetchAssocForWindow(rows, wD11),
          fetchAssocForWindow(rows, w618),
        ]);
        setAssocD11(r1);
        setAssoc618(r2);
      } catch (e: any) {
        setAssocD11([]);
        setAssoc618([]);
        console.error('assoc_window failed:', e?.message || e);
      }
    })();
  }, [rows]);

  const monthly12 = useMemo(() => {
    if (!monthlyAll.length) return [];
    const lastMonth = monthlyAll[monthlyAll.length-1].yyyymm + '-01';
    const cutoff = subMonths(new Date(lastMonth), 11);
    return monthlyAll.filter(m => new Date(m.yyyymm + '-01') >= cutoff);
  }, [monthlyAll]);

  const kpi = useMemo(() => {
    const revenue = rows.reduce((s,r)=>s+r.revenue,0);
    const orders = rows.length;
    const customers = new Set(rows.map(r=>r.customer_email || r.customer_name)).size;
    const aov = orders ? revenue/orders : 0;
    return { revenue, orders, customers, aov };
  }, [rows]);

  const spark5 = useMemo(() => monthlyAll.slice(-5).map(m => ({ x: m.yyyymm, y: m.revenue })), [monthlyAll]);

  const momDetail = useMemo(() => {
    if (monthlyAll.length < 2) return null;
    const a = monthlyAll[monthlyAll.length-2];
    const b = monthlyAll[monthlyAll.length-1];
    const pct = a.revenue===0?null:(b.revenue-a.revenue)/a.revenue;
    return {
      pct, diff: b.revenue-a.revenue, prev:a.revenue, curr:b.revenue,
      labels: { prev: a.yyyymm, curr: b.yyyymm }
    };
  }, [monthlyAll]);

  const wowDetail = useMemo(() => {
    if (rows.length===0) return null;
    const dates = rows.map(r=>parseISO(r.date)).sort((a,b)=>a.getTime()-b.getTime());
    const last = dates[dates.length-1];
    const w2_start = subDays(last, 6), w1_start = subDays(w2_start, 7);
    let r1=0, r2=0;
    for (const r of rows) {
      const d=parseISO(r.date);
      if (d>=w1_start && d<w2_start) r1+=r.revenue;
      else if (d>=w2_start && d<=last) r2+=r.revenue;
    }
    const prevLabel = `${format(w1_start,'yyyy-MM-dd')} ~ ${format(subDays(w2_start,1),'yyyy-MM-dd')}`;
    const currLabel = `${format(w2_start,'yyyy-MM-dd')} ~ ${format(last,'yyyy-MM-dd')}`;
    return { pct: r1===0?null:(r2-r1)/r1, diff: r2-r1, prev:r1, curr:r2, labels:{prev:prevLabel, curr:currLabel} };
  }, [rows]);

  const top5: TopItem[] = useMemo(() => {
    const m = new Map<string, number>();
    for (const r of rows) m.set(r.product_name, (m.get(r.product_name) || 0) + r.revenue);
    return [...m.entries()].map(([product_name, revenue]) => ({ product_name, revenue }))
      .sort((a,b)=>b.revenue-a.revenue).slice(0,5);
  }, [rows]);

  const topCats: TopCat[] = useMemo(() => {
    const m = new Map<string, number>();
    for (const r of rows) {
      const cat = (r.product_category && r.product_category.trim()) ? r.product_category : '未分類';
      m.set(cat, (m.get(cat)||0) + r.revenue);
    }
    return [...m.entries()].map(([category, revenue])=>({category, revenue}))
      .sort((a,b)=>b.revenue-a.revenue).slice(0,5);
  }, [rows]);

  const discountTop5ByQty = useMemo(() => {
    if (!rows.length) return [];
    const m = new Map<string, { fullQty: number; discQty: number; total: number }>();
    for (const r of rows) {
      if ((r.revenue ?? 0) <= 0) continue;
      const rec = m.get(r.product_name) || { fullQty: 0, discQty: 0, total: 0 };
      if (r.was_discounted) rec.discQty += r.quantity || 1;
      else rec.fullQty += r.quantity || 1;
      rec.total += r.quantity || 1;
      m.set(r.product_name, rec);
    }
    return [...m.entries()]
      .map(([name, v]) => ({ product_name: name, ...v }))
      .sort((a,b)=>b.total - a.total)
      .slice(0, 5);
  }, [rows]);

  const mustDiscountTop5 = useMemo(() => {
    const MIN_TOTAL = 5;
    const m = new Map<string, { fullQty: number; discQty: number; total: number }>();
    for (const r of rows) {
      if ((r.revenue ?? 0) <= 0) continue;
      const rec = m.get(r.product_name) || { fullQty: 0, discQty: 0, total: 0 };
      if (r.was_discounted) rec.discQty += r.quantity || 1;
      else rec.fullQty += r.quantity || 1;
      rec.total += r.quantity || 1;
      m.set(r.product_name, rec);
    }
    return [...m.entries()]
      .map(([name, v]) => ({
        product_name: name,
        ...v,
        ratio: v.total>0 ? v.discQty/v.total : 0
      }))
      .filter(x => x.total >= MIN_TOTAL && x.discQty > 0)
      .sort((a,b)=> b.discQty - a.discQty)
      .slice(0, 5);
  }, [rows]);

  type CustAgg = { id: string; last: Date; orders: number; revenue: number };
  const customersAgg: CustAgg[] = useMemo(() => {
    if (!rows.length) return [];
    const map = new Map<string, { last: Date; ordersSet: Set<string>; revenue: number }>();
    for (const r of rows) {
      const id = (r.customer_email && r.customer_email.trim()) || (r.customer_name && r.customer_name.trim()) || '';
      if (!id) continue;
      const keyOrder = r.order_id ? r.order_id : `${id}_${r.date}`;
      const rec = map.get(id) || { last: new Date(0), ordersSet: new Set<string>(), revenue: 0 };
      const d = parseISO(r.date);
      if (d > rec.last) rec.last = d;
      rec.ordersSet.add(keyOrder);
      rec.revenue += r.revenue;
      map.set(id, rec);
    }
    return [...map.entries()].map(([id, v]) => ({ id, last: v.last, orders: v.ordersSet.size, revenue: v.revenue }));
  }, [rows]);

  // RFM 等級
  const localRfmLevels = useMemo(() => {
    if (!customersAgg.length) return new Map<string,'low'|'mid'|'high'>();
    const maxDate = rows.map(r=>parseISO(r.date)).sort((a,b)=>b.getTime()-a.getTime())[0] || new Date();
    const recDays = customersAgg.map(c => ({ id: c.id, r: Math.max(0, (maxDate.getTime() - c.last.getTime())/86400000), o: c.orders, m: c.revenue }));
    const rank = (arr: number[]) => {
      const sorted = [...arr].slice().sort((a,b)=>a-b);
      const pos = (v: number) => (sorted.findIndex(x=>x===v) + sorted.lastIndexOf(v)) / 2 + 1;
      return arr.map(v => pos(v)/sorted.length);
    };
    const rRank = rank(recDays.map(x=>-x.r));
    const fRank = rank(recDays.map(x=> x.o));
    const mRank = rank(recDays.map(x=> x.m));
    const score = recDays.map((x,i)=> ({ id: x.id, s: 0.2*rRank[i] + 0.3*fRank[i] + 0.5*mRank[i] }));
    const sArr = score.map(x=>x.s).sort((a,b)=>a-b);
    const t1 = sArr[Math.floor(sArr.length/3)] ?? 0.33;
    const t2 = sArr[Math.floor(sArr.length*2/3)] ?? 0.66;
    const lv = new Map<string,'low'|'mid'|'high'>();
    score.forEach(x=>{
      if (x.s <= t1) lv.set(x.id,'low');
      else if (x.s <= t2) lv.set(x.id,'mid');
      else lv.set(x.id,'high');
    });
    return lv;
  }, [customersAgg, rows]);

  // 平均頻次
  const avgFreqByLevel = useMemo(() => {
    const out = { low: 0, mid: 0, high: 0 };
    const cnt = { low: 0, mid: 0, high: 0 };
    customersAgg.forEach(c=>{
      const lv = localRfmLevels.get(c.id) || 'mid';
      out[lv] += c.orders;
      cnt[lv] += 1;
    });
    return {
      low: cnt.low ? out.low/cnt.low : 0,
      mid: cnt.mid ? out.mid/cnt.mid : 0,
      high: cnt.high ? out.high/cnt.high : 0,
    };
  }, [customersAgg, localRfmLevels]);

  // 高價值名單
  const highValueDownloadRows = useMemo(()=>{
    const arr = customersAgg
      .filter(c => (localRfmLevels.get(c.id) || 'mid') === 'high')
      .map(c => ({
        customer_email: c.id,
        last_order_date: format(c.last, 'yyyy-MM-dd'),
        lifetime_orders: c.orders,
        lifetime_revenue: Math.round(c.revenue),
      }))
      .sort((a,b)=> b.lifetime_revenue - a.lifetime_revenue)
      .slice(0, 500);
    return arr;
  }, [customersAgg, localRfmLevels]);

  /* ===== 城市 Top10（依營收） ===== */
  const cityTop10 = useMemo(()=>{
    const m = new Map<string, number>();
    for (const r of rows) {
      const city = (r.shipping_city || '').trim();
      if (!city) continue;
      m.set(city, (m.get(city)||0) + (r.revenue || 0));
    }
    return [...m.entries()]
      .map(([city, revenue])=>({ city, revenue }))
      .sort((a,b)=> b.revenue - a.revenue)
      .slice(0,10);
  }, [rows]);

  const cityAxisWidth = useMemo(() => {
    const maxLen = cityTop10.reduce((m, x) => Math.max(m, (x.city || '').length), 0);
    return Math.min(240, Math.max(110, Math.ceil(maxLen * 8.5)));
  }, [cityTop10]);

  /* ===== 檔期名單：去年雙11 / 去年黑五 / 618===== */
  function getLastYearKeyDate() {
    if (!rows.length) return new Date();
    const last = rows.map(r=>parseISO(r.date)).sort((a,b)=>b.getTime()-a.getTime())[0];
    const y = (last?.getFullYear() || new Date().getFullYear()) - 1;
    return y;
  }
  function double11Window(year: number) {
    const s = startOfDay(new Date(year, 10, 1));
    const e = endOfDay(new Date(year, 10, 15));
    return { s, e };
  }
  function blackFridayWindow(year: number) {
    const nov1 = new Date(year, 10, 1);
    const day = nov1.getDay();
    const firstFri = 5 - day >= 0 ? 5 - day : 12 - day;
    const friday4th = new Date(year, 10, 1 + firstFri + 7*3);
    const s = startOfDay(friday4th);
    const e = endOfDay(new Date(friday4th.getFullYear(), friday4th.getMonth(), friday4th.getDate()+3));
    return { s, e };
  }  

  function six18Window(year: number) {
    const s = startOfDay(new Date(year, 5, 1));
    const e = endOfDay(new Date(year, 5, 18));
    return { s, e };
  }  
    
  function buildCampaignTop(year: number, win: {s: Date; e: Date}) {
    const byCust: Record<string,{ orders:number; rev:number; ordersIn:number; revIn:number; last:Date }> = {};
    rows.forEach(r=>{
      const id = (r.customer_email && r.customer_email.trim()) || (r.customer_name && r.customer_name.trim()) || '';
      if (!id) return;
      const d = parseISO(r.date);
      const inWin = d >= win.s && d <= win.e;
      const rec = byCust[id] || { orders:0, rev:0, ordersIn:0, revIn:0, last: new Date(0) };
      rec.orders += 1; rec.rev += r.revenue;
      if (inWin) { rec.ordersIn += 1; rec.revIn += r.revenue; }
      if (d > rec.last) rec.last = d;
      byCust[id] = rec;
    });
    const list = Object.entries(byCust).map(([id,v])=>{
      const avgOrder = v.orders ? v.rev / v.orders : 0;
      return { customer_email: id, amount: Math.round(v.revIn), avgOrder, last: v.last };
    })
    .filter(x => x.amount > x.avgOrder)
    .sort((a,b)=> b.amount - a.amount)
    .slice(0, 10);
    return list;
  }
  const campaignLists = useMemo(()=>{
    if (!rows.length) return { d11: [], six18: [] as {customer_email:string; amount:number; avgOrder:number; last:Date}[] };
    const y = getLastYearKeyDate();
    const d11 = buildCampaignTop(y, double11Window(y));
    const six18  = buildCampaignTop(y, six18Window(y));
    return { d11, six18 };
  }, [rows]);

  const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL ??
    (process.env.NODE_ENV === 'production'
      ? 'https://business-analysis-model-backend.onrender.com'
      : 'http://127.0.0.1:8000');

  async function fetchWithRetry(url: string, opts: RequestInit = {}, tries = 4) {
    for (let i = 0; i < tries; i++) {
      const ctrl = new AbortController();
      const timeout = setTimeout(() => ctrl.abort(), 45_000);
      try {
        const res = await fetch(url, { ...opts, signal: ctrl.signal });
        if (res.ok) return res;
        if ([429, 500, 502, 503, 504, 522, 524].includes(res.status)) throw new Error('temporary');
        return res;
      } catch {
        await new Promise(r => setTimeout(r, Math.min(16_000, 2 ** i * 1000)));
      } finally {
        clearTimeout(timeout);
      }
    }
    throw new Error('backend unavailable');
  }

  async function fetchAssocForWindow(rowsInput: Row[], win: { s: Date; e: Date }) {
    const payload = {
      rows: rowsInput.map(r => ({
        date: r.date,
        product_name: r.product_name,
        revenue: r.revenue,
        quantity: r.quantity,
        product_category: r.product_category ?? undefined,
        customer_name: r.customer_name ?? undefined,
        customer_email: r.customer_email ?? undefined,
        order_id: r.order_id ?? undefined,
      })),
      date_from: format(win.s, 'yyyy-MM-dd'),
      date_to: format(win.e, 'yyyy-MM-dd'),
    };

    const res = await fetchWithRetry(`${API_BASE}/assoc_window`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return (data?.assoc_rules ?? []) as AssocRule[];
  }

  async function analyzeOnServer(input?: Row[]) {
    const slimRows = (input ?? rows).map(r => ({
      date: r.date,
      product_name: r.product_name,
      revenue: r.revenue,
      quantity: r.quantity,
      product_category: r.product_category ?? undefined,
      customer_name: r.customer_name ?? undefined,
      customer_email: r.customer_email ?? undefined,
      order_id: r.order_id ?? undefined,
    }));

    const payload = { rows: slimRows };
    if (!payload.rows.length) return;

    setServerNote('喚醒後端/分析中…（首次呼叫可能較久）');

    try {
      const res = await fetchWithRetry(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status} ${text || res.statusText}`);
      }

      const data: ServerInsights = await res.json();
      setServer(data);

      const msgs: string[] = [];
      if (data.diagnostics?.assoc && !data.diagnostics.assoc.ok) {
        const d = data.diagnostics.assoc;
        if (d.reason === 'single_item_baskets_or_too_few') msgs.push('關聯規則：多數訂單為單一品項或有效籃子太少。');
        if (d.reason === 'too_sparse_support') msgs.push('關聯規則：組合過於稀疏，建議放寬門檻或拉長期間。');
        if (d.reason === 'no_significant_rules') msgs.push('關聯規則：未找到顯著規則（lift/信賴度不足）。');
        if (d.reason === 'algorithm_error') msgs.push('關聯規則：演算法錯誤，請稍後重試。');
      }
      if (data.diagnostics?.rfm && !data.diagnostics.rfm.ok) {
        const r = data.diagnostics.rfm;
        if (r.reason === 'no_customer_key_or_too_few') msgs.push('RFM：缺少可辨識的客戶鍵（email/name）或客戶數過少（<3）。');
        if (r.reason === 'algorithm_error') msgs.push('RFM：演算法錯誤，請稍後重試。');
      }
      setServerNote(msgs.join('  ') || '');
    } catch (e: any) {
      setServerNote('呼叫後端失敗：' + (e?.message || 'unknown'));
    }
  }

  /* ====== 產出簡報（下載 .pptx）====== */
  async function downloadDeck() {
    if (!rows.length) {
      setNotice({ level: 'warn', msg: '請先上傳 CSV 才能產生簡報' });
      return;
    }

    setDeckBusy(true);
    setDeckMsg('準備產生簡報…');

    try {
      // 1) 若沒有現成的 insights，就即時呼叫 /analyze 拿一次
      let insights: any = server;
      if (!insights) {
        setDeckMsg('後端分析中…');
        const slimRows = rows.map(r => ({
          date: r.date,
          product_name: r.product_name,
          revenue: r.revenue,
          quantity: r.quantity,
          product_category: r.product_category ?? undefined,
          customer_name: r.customer_name ?? undefined,
          customer_email: r.customer_email ?? undefined,
          order_id: r.order_id ?? undefined,
        }));

        const resAnalyze = await fetchWithRetry(`${API_BASE}/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
          body: JSON.stringify({ rows: slimRows }),
        });
        if (!resAnalyze.ok) throw new Error(`分析失敗（HTTP ${resAnalyze.status}）`);
        insights = await resAnalyze.json();
      }

      // 2) 送到 /generate_deck 產出 PPTX
      setDeckMsg('AI 規劃投影片/產出中…');
      const resDeck = await fetchWithRetry(`${API_BASE}/generate_deck`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
        body: JSON.stringify({
          title: 'Business Product Analysis',
          insights, // 後端會自己挑需要的欄位
        }),
      });
      if (!resDeck.ok) {
        const t = await resDeck.text().catch(() => '');
        throw new Error(`HTTP ${resDeck.status}${t ? ` ｜${t.slice(0, 200)}` : ''}`);
      }

      // 3) 下載檔案
      const blob = await resDeck.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Business_Analysis_${new Date().toISOString().slice(0,10)}.pptx`;
      a.click();
      URL.revokeObjectURL(url);

      setDeckMsg('已下載簡報');
    } catch (e: any) {
      setDeckMsg('產生失敗：' + (e?.message || 'unknown'));
    } finally {
      setDeckBusy(false);
    }
  }



  /* == UI == */
  return (
    <>
      {/* ===== Full-bleed Hero ===== */}
      <header className="w-full">
        <div className="relative h-44 md:h-56 bg-slate-700">
          <div className="absolute inset-0 bg-gradient-to-tr from-slate-700 to-sky-700/30" />
          <div className="relative h-full max-w-7xl mx-auto px-6 md:px-8 flex items-center">
            <div>
              <h1 className="text-3xl md:text-4xl font-extrabold tracking-wide text-white">
                BUSINESS PRODUCT ANALYSIS
              </h1>
              <p className="mt-2 text-slate-200 text-sm md:text-base">
                上傳銷售資料，立即取得趨勢、分群與關聯規則洞察
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="min-h-screen bg-white text-slate-800">
        <div className="max-w-7xl mx-auto px-6 md:px-8">
          {/* 上傳卡片吃進 Hero 一點 */}
          <div className="relative z-10 -mt-6 md:-mt-8 mb-4">
            <div className="border border-slate-200 rounded-2xl p-4 bg-gradient-to-b from-slate-50 to-white shadow-lg">
              <label className="block text-sm font-medium mb-2 text-slate-600">
                上傳 CSV（需含：date / product_name / revenue；常見別名會自動辨識）
              </label>
              <input
                type="file" accept=".csv"
                onChange={(e)=> e.target.files?.[0] && onFile(e.target.files[0])}
                className="block w-full text-sm file:mr-4 file:rounded-lg file:border-0
                           file:bg-slate-100 file:text-slate-700 file:px-4 file:py-2 file:cursor-pointer"
              />
              {fileName && <p className="text-xs text-slate-500 mt-2">已選擇：{fileName}</p>}
            </div>
          </div>

          <div className="flex items-center justify-between mb-6">
            <div className="flex gap-3">
              <TabBtn active={tab==='summary'}   onClick={()=>setTab('summary')}>Data Summary</TabBtn>
              <TabBtn active={tab==='customers'} onClick={()=>setTab('customers')}>Customer Analysis</TabBtn>
              <TabBtn active={tab==='products'}  onClick={()=>setTab('products')}>Product Analysis</TabBtn>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={downloadDeck}
                disabled={deckBusy || rows.length === 0}
                className={`px-4 py-2 rounded-xl border shadow-sm transition ${
                  deckBusy || rows.length === 0
                    ? 'bg-slate-100 text-slate-400 border-slate-200 cursor-not-allowed'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                }`}
                title={rows.length === 0 ? '請先上傳資料' : '匯出 PowerPoint'}
              >
                {deckBusy ? '產生簡報中…' : '產出 PPT 簡報'}
              </button>

              {deckMsg && <span className="text-xs text-slate-500">{deckMsg}</span>}
            </div>
          </div>

          {tab==='summary' && (
            <section className="space-y-6 mb-12">
              {/* KPI */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <KpiCardWithSpark title="總營收" value={formatCurrency(kpi.revenue)} data={spark5}/>
                <KpiCardWithSpark title="訂單數" value={kpi.orders.toLocaleString()} data={spark5}/>
                <KpiCardWithSpark title="客戶數" value={kpi.customers.toLocaleString()} data={spark5}/>
                <KpiCardWithSpark title="客單價" value={formatCurrency(kpi.aov)} data={spark5}/>
              </div>

              {/* 趨勢 + MoM/WoW */}
              <div className="grid md:grid-cols-3 gap-6 items-stretch">
                <div className="md:col-span-2 rounded-2xl border border-slate-200 p-4">
                  <div className="flex items-end justify-between mb-2">
                    <h2 className="text-base font-semibold text-slate-700">每月營收（近一年）</h2>
                  </div>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={monthly12}>
                        <defs>
                          <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.35} />
                            <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="yyyymm" tick={{ fill: '#475569' }} />
                        <YAxis tick={{ fill: '#475569' }} tickFormatter={(v:number)=> formatCurrency(v)} width={72} />
                        <Tooltip
                          formatter={(v:any)=>[formatCurrency(v as number), 'revenue']}
                          labelFormatter={(l:any)=> String(l)}
                        />
                        <Area type="monotone" dataKey="revenue" stroke="#2563eb" strokeWidth={2} fill="url(#revGrad)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="grid grid-rows-2 gap-6">
                  <DeltaCard label="MoM" detail={momDetail}/>
                  <DeltaCard label="WoW" detail={wowDetail}/>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="rounded-2xl border border-slate-200 p-4">
                  <h2 className="text-base font-semibold mb-2 text-slate-700">Top 5 產品（依營收）</h2>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left border-b border-slate-200 text-slate-500">
                        <th className="py-2 w-10">#</th><th className="py-2">產品名稱</th><th className="py-2 text-right">營收</th>
                      </tr>
                    </thead>
                    <tbody>
                      {top5.map((t, i) => (
                        <tr key={t.product_name} className="border-b border-slate-100 last:border-none">
                          <td className="py-2">{i + 1}. </td>
                          <td className="py-2">{t.product_name}</td>
                          <td className="py-2 text-right">{formatCurrency(t.revenue)}</td>
                        </tr>
                      ))}
                      {top5.length===0 && <tr><td colSpan={3} className="py-6 text-center text-slate-400">請先上傳 CSV 檔</td></tr>}
                    </tbody>
                  </table>
                </div>

                <div className="rounded-2xl border border-slate-200 p-4">
                  <h2 className="text-base font-semibold mb-2 text-slate-700">Top 5 類別（依營收）</h2>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={topCats} layout="vertical" margin={{ left: 24, right: 96, top: 8, bottom: 8 }}>
                        <XAxis type="number" hide />
                        <YAxis type="category" dataKey="category" width={140} tick={{ fill: '#475569' }} />
                        <Tooltip formatter={(v:any) => [formatCurrency(Math.round(v as number)), 'revenue']} labelFormatter={(l:any) => String(l)} />
                        <defs>
                          <linearGradient id="barGrad" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.6} />
                            <stop offset="100%" stopColor="#2563eb" stopOpacity={0.9} />
                          </linearGradient>
                        </defs>
                        <Bar dataKey="revenue" radius={[12,12,12,12]} fill="url(#barGrad)">
                          <LabelList dataKey="revenue" position="right" offset={8} formatter={(v:any)=> formatCurrency(Math.round(v))} />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="rounded-2xl border border-slate-200 p-4">
                  <h2 className="text-base font-semibold mb-2 text-slate-700">前 5 暢銷（量）｜正價 vs 折扣</h2>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={discountTop5ByQty.map(d=>({ name: d.product_name, 正價: d.fullQty, 折扣: d.discQty }))}
                        layout="horizontal"
                        margin={{ left: 8, right: 16, top: 8, bottom: 8 }}
                      >
                        <XAxis dataKey="name" tick={{ fill: '#475569' }} hide />
                        <YAxis tick={{ fill: '#475569' }} />
                        <Tooltip formatter={(v:any, k:any)=> [Number(v).toLocaleString(), k]} />
                        <defs>
                          <linearGradient id="fullGrad" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0%" stopColor="#93c5fd" />
                            <stop offset="100%" stopColor="#3b82f6" />
                          </linearGradient>
                          <linearGradient id="discGrad" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0%" stopColor="#fda4af" />
                            <stop offset="100%" stopColor="#ef4444" />
                          </linearGradient>
                        </defs>
                        <Bar dataKey="正價" fill="url(#fullGrad)" radius={[8,8,0,0]} />
                        <Bar dataKey="折扣" fill="url(#discGrad)" radius={[8,8,0,0]} />
                        <LabelList dataKey="正價" position="top" formatter={(v:any)=>Number(v).toLocaleString()} />
                        <LabelList dataKey="折扣" position="top" formatter={(v:any)=>Number(v).toLocaleString()} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">＊折扣偵測：有折扣金額、或原價 &gt; 成交單價、或有優惠碼。</p>
                </div>

                <div className="rounded-2xl border border-slate-200 p-4">
                  <h2 className="text-base font-semibold mb-2 text-slate-700">最仰賴折扣的 Top 5（以折扣銷量占比）</h2>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left border-b border-slate-200 text-slate-500">
                        <th className="py-2">產品</th>
                        <th className="py-2 text-right">折扣量</th>
                        <th className="py-2 text-right">正價量</th>
                        <th className="py-2 text-right">折扣占比</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mustDiscountTop5.map((r)=>(
                        <tr key={r.product_name} className="border-b border-slate-100 last:border-none">
                          <td className="py-2">{r.product_name}</td>
                          <td className="py-2 text-right">{r.discQty.toLocaleString()}</td>
                          <td className="py-2 text-right">{r.fullQty.toLocaleString()}</td>
                          <td className="py-2 text-right">{(r.ratio*100).toFixed(1)}%</td>
                        </tr>
                      ))}
                      {mustDiscountTop5.length===0 && <tr><td colSpan={4} className="py-6 text-center text-slate-400">尚無可辨識的折扣紀錄</td></tr>}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          )}

          {/* ====== Customers ======= */}
          {tab==='customers' && (
            <section className="space-y-6 mb-12">
              {/* RFM 佔比 + 下載 */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-base font-semibold text-slate-700">客群價值佔比（RFM × KMeans）</h2>
                  <div className="flex gap-2">
                    {server?.nurture_mid && server.nurture_mid.length>0 && (
                      <button
                        onClick={()=>downloadCsv('nurture_mid.csv', server.nurture_mid!)}
                        className="px-3 py-2 rounded-md border border-slate-300 text-slate-700 hover:bg-slate-50 shadow-sm">
                        下載中價值客戶名單
                      </button>
                    )}
                    {highValueDownloadRows.length>0 && (
                      <button
                        onClick={()=>downloadCsv('high_value.csv', highValueDownloadRows)}
                        className="px-3 py-2 rounded-md border border-slate-300 text-slate-700 hover:bg-slate-50 shadow-sm">
                        下載高價值顧客名單
                      </button>
                    )}
                  </div>
                </div>
                <div className="grid md:grid-cols-3 gap-4">
                  <SegmentCard title="高價值" share={server?.rfm_share?.high ?? 0} count={server?.rfm_counts?.high} avg={server?.rfm_avg_monetary?.high} gradientFrom="#60a5fa" gradientTo="#2563eb" />
                  <SegmentCard title="中價值" share={server?.rfm_share?.mid ?? 0} count={server?.rfm_counts?.mid}  avg={server?.rfm_avg_monetary?.mid}  gradientFrom="#93c5fd" gradientTo="#60a5fa" />
                  <SegmentCard title="低價值" share={server?.rfm_share?.low ?? 0} count={server?.rfm_counts?.low}  avg={server?.rfm_avg_monetary?.low}  gradientFrom="#cbd5e1" gradientTo="#94a3b8" />
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  佔比為「營收占比」。人數與平均金額來自後端 RFM 分群結果（不足 3 位客戶時可能無法計算）。
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="rounded-2xl border border-slate-200 p-4 shadow-sm">
                  <h3 className="text-sm font-semibold text-slate-700 mb-2">高/中/低價值 各自平均消費金額</h3>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left border-b border-slate-200 text-slate-500">
                        <th className="py-2">群組</th><th className="py-2 text-right">平均金額</th>
                      </tr>
                    </thead>
                    <tbody>
                      {['high','mid','low'].map((k)=>(
                        <tr key={k} className="border-b border-slate-100 last:border-none">
                          <td className="py-2">{labelOf(k)}</td>
                          <td className="py-2 text-right">{formatCurrency((server?.rfm_avg_monetary as any)?.[k] ?? 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="rounded-2xl border border-slate-200 p-4 shadow-sm">
                  <h3 className="text-sm font-semibold text-slate-700 mb-2">高/中/低價值 各自平均消費頻次</h3>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left border-b border-slate-200 text-slate-500">
                        <th className="py-2">群組</th><th className="py-2 text-right">平均頻次（單位：筆）</th>
                      </tr>
                    </thead>
                    <tbody>
                      {['high','mid','low'].map((k)=>(
                        <tr key={k} className="border-b border-slate-100 last:border-none">
                          <td className="py-2">{labelOf(k)}</td>
                          <td className="py-2 text-right">{avgFreqByLevel[k as 'high'|'mid'|'low'].toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="text-xs text-slate-500 mt-2">＊頻次為前端近似估計（orders 計數），與後端 KMeans 分群不一定完全相同。</p>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-200 p-4 shadow-sm">
                <h3 className="text-sm font-semibold text-slate-700 mb-2">消費最高城市 Top10（依營收）</h3>
                <ResponsiveContainer
                  width="100%"
                  height={Math.max(280, cityTop10.length * 44)}
                >
                  <BarChart
                    data={cityTop10}
                    layout="vertical"
                    margin={{ left: 12, right: 96, top: 12, bottom: 12 }}
                    barCategoryGap={12}
                  >
                    <XAxis type="number" hide />
                    <YAxis
                      type="category"
                      dataKey="city"
                      width={cityAxisWidth}
                      interval={0}
                      tick={{ fontSize: 12, fill: '#475569' }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip formatter={(v:any)=>[formatCurrency(Math.round(v as number)), 'revenue']} />
                    <defs>
                      <linearGradient id="cityBarGrad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopOpacity={0.35} stopColor="#86efac" />
                        <stop offset="100%" stopOpacity={0.95} stopColor="#10b981" />
                      </linearGradient>
                    </defs>
                    <Bar dataKey="revenue" radius={[12,12,12,12]} fill="url(#cityBarGrad)">
                      <LabelList
                        dataKey="revenue"
                        position="right"
                        offset={8}
                        formatter={(v:any)=> formatCurrency(Math.round(v))}
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <CampaignCard
                  title="去年雙11消費高於個人平均的客戶（Top10）"
                  rows={campaignLists.d11}
                  downloadName="double11_high.csv"
                />
                <CampaignCard
                  title="去年618年中慶消費高於個人平均的客戶（Top10）"
                  rows={campaignLists.six18}
                  downloadName="618_high.csv"
                />
              </div>
            </section>
          )}

          {/* ==== Products ==== */}
          {tab==='products' && (
            <section className="space-y-6 mb-12">
              {/* 潛力商品 */}
              <div className="rounded-2xl border border-slate-200 p-4 shadow-sm">
                <h2 className="text-base font-semibold mb-2 text-slate-700">潛力商品（近 3 月成長 & 非 Top5）</h2>
                <table className="w-full text-sm">
                  <thead><tr className="text-left border-b border-slate-200 text-slate-500">
                    <th className="py-2">產品</th>
                    <th className="py-2 text-right">最近 MoM</th>
                    <th className="py-2 text-right">近三月營收</th>
                  </tr></thead>
                  <tbody>
                    {server?.potential_products?.map((p)=>(
                      <tr key={p.product_name} className="border-b border-slate-100 last:border-none">
                        <td className="py-2">{p.product_name}</td>
                        <td className="py-2 text-right">{(p.mom_last*100).toFixed(1)}%</td>
                        <td className="py-2 text-right">{formatCurrency(p.sum3_revenue)}</td>
                      </tr>
                    ))}
                    {(!server || server.potential_products.length===0) && <tr><td colSpan={3} className="py-6 text-center text-slate-400">暫無候選</td></tr>}
                  </tbody>
                </table>
              </div>

              <div className="rounded-2xl border border-slate-200 overflow-hidden shadow-sm">
                <div className="grid grid-cols-12 bg-slate-50 text-slate-600 text-sm font-medium px-4 py-3">
                  <div className="col-span-5">若顧客購買</div>
                  <div className="col-span-2 text-center"> </div>
                  <div className="col-span-5">也常一起買</div>
                </div>
                {server?.assoc_rules?.map((r,i)=>(
                  <div key={i} className="grid grid-cols-12 items-center border-t border-slate-100 px-4 py-3">
                    <div className="col-span-5 text-slate-800 text-sm">{r.antecedents.join(' + ')}</div>
                    <div className="col-span-2 text-center text-slate-400">→</div>
                    <div className="col-span-5 text-slate-800 text-sm">{r.consequents.join(' + ')}</div>

                    <div className="col-span-12 flex gap-4 mt-2 text-xs text-slate-600">
                      <Badge label="Lift" value={r.lift.toFixed(2)} />
                      <Badge label="Confidence" value={(r.confidence*100).toFixed(1)+'%'} />
                      <Badge label="Support" value={(r.support*100).toFixed(1)+'%'} />
                    </div>
                  </div>
                ))}
                {(!server || server.assoc_rules.length===0) && (
                  <div className="p-6 text-center text-slate-400">暫無顯著規則</div>
                )}
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <SuggestionCard
                  title="雙11檔期建議（Bundle & 檔期）"
                  rules={assocD11 || []}
                  y={getLastYearKeyDate()}
                  winBuilder={double11Window}
                  monthlyAll={monthlyAll}
                />
                <SuggestionCard
                  title="618檔期建議（Bundle & 檔期）"
                  rules={assoc618 || []}
                  y={getLastYearKeyDate()}
                  winBuilder={six18Window}
                  monthlyAll={monthlyAll}
                />
              </div>
            </section>
          )}

          {serverNote && (
            <div className="mb-12 rounded-md border px-3 py-2 text-sm border-amber-300 bg-amber-50 text-amber-800">
              {serverNote}
            </div>
          )}
        </div>
      </main>
    </>
  );
}
// ===== Currency & small utils (SSR-safe) =====
const LOCALE = process.env.NEXT_PUBLIC_LOCALE || 'zh-TW';
const CURRENCY = process.env.NEXT_PUBLIC_CURRENCY || 'TWD';

export function formatCurrency(
  v: number | null | undefined,
  options: Intl.NumberFormatOptions = {}
): string {
  const n = Number(v ?? 0);
  try {
    return new Intl.NumberFormat(LOCALE, {
      style: 'currency',
      currency: CURRENCY,
      maximumFractionDigits: 0,
      ...options,
    }).format(n);
  } catch {
    return `${n.toLocaleString()} ${CURRENCY}`;
  }
}

function labelOf(k: 'high' | 'mid' | 'low') {
  return k === 'high' ? '高價值' : k === 'mid' ? '中價值' : '低價值';
}

function downloadCsv(fileName: string, rows: any[]) {
  if (!rows || rows.length === 0) return;
  const header = Object.keys(rows[0]);
  const lines = [
    header.join(','),
    ...rows.map((r) =>
      header
        .map((h) => {
          const raw = r[h] ?? '';
          const s = String(raw).replace(/"/g, '""');
          return /[",\n]/.test(s) ? `"${s}"` : s;
        })
        .join(',')
    ),
  ];
  const blob = new Blob([lines.join('\n')], {
    type: 'text/csv;charset=utf-8;',
  });
  if (typeof window === 'undefined') return; 
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  a.click();
  URL.revokeObjectURL(url);
}

/* ====== 小元件 ====== */
function TabBtn({ active, onClick, children }:{active:boolean; onClick:()=>void; children:any}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-xl border ${
        active ? 'bg-sky-600 text-white border-sky-700 shadow-md' : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
      }`}
    >
      {children}
    </button>
  );
}

function KpiCardWithSpark({
  title,
  value,
  data,
}: {
  title: string;
  value: string;
  data: { x: string; y: number }[];
}) {
  return (
    <div className="rounded-2xl border border-slate-200 p-4 bg-white shadow-sm">
      <div className="text-xs text-slate-500 mb-1">{title}</div>
      <div className="text-xl font-semibold text-slate-800">{value}</div>
      <div className="h-12 mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.5} />
                <stop offset="100%" stopColor="#60a5fa" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <XAxis dataKey="x" hide />
            <YAxis hide />
            <Area
              type="monotone"
              dataKey="y"
              stroke="#60a5fa"
              strokeWidth={1}
              fill="url(#sparkGrad)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
function Badge({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-slate-300 px-2 py-0.5 text-xs text-slate-600">
      <span className="opacity-70">{label}:</span>
      <span className="font-medium">{value}</span>
    </span>
  );
}
type DeltaDetail = {
  pct: number | null;
  diff: number;
  prev: number;
  curr: number;
  labels: { prev: string; curr: string };
} | null;

function DeltaCard({ label, detail }: { label: string; detail: DeltaDetail }) {
  const has = !!detail && detail.pct !== null;
  const pct = has ? (detail!.pct! * 100) : null;
  const up = (pct ?? 0) >= 0;

  return (
    <div className="rounded-2xl border border-slate-200 p-4 shadow-sm">
      <div className="text-sm font-semibold text-slate-700 mb-2">{label}</div>
      {has ? (
        <div>
          <div className={`text-2xl font-bold ${up ? 'text-emerald-600' : 'text-rose-600'}`}>
            {up ? '▲' : '▼'} {Math.abs(pct!).toFixed(1)}%
          </div>
          <div className="mt-1 text-xs text-slate-500">
            {detail!.labels.prev}: {formatCurrency(detail!.prev)} → {detail!.labels.curr}: {formatCurrency(detail!.curr)}
          </div>
          <div className="mt-1 text-xs text-slate-500">
            差額：{formatCurrency(detail!.diff)}
          </div>
        </div>
      ) : (
        <div className="text-slate-400 text-sm">資料不足</div>
      )}
    </div>
  );
}

/* == RFM 區塊卡片 == */
function SegmentCard({
  title,
  share,
  count,
  avg,
  gradientFrom,
  gradientTo,
}: {
  title: string;
  share: number;
  count?: number;
  avg?: number;
  gradientFrom: string;
  gradientTo: string;
}) {
  const pct = Math.max(0, Math.min(1, Number(share || 0)));
  return (
    <div className="rounded-2xl border border-slate-200 p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-slate-700">{title}</div>
        <div className="text-slate-500 text-xs">{(pct * 100).toFixed(1)}%</div>
      </div>
      <div className="mt-2 h-2 w-full rounded-full bg-slate-100 overflow-hidden">
        <div
          className="h-full"
          style={{
            width: `${pct * 100}%`,
            background: `linear-gradient(90deg, ${gradientFrom}, ${gradientTo})`,
          }}
        />
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-600">
        <div className="flex justify-between">
          <span>人數</span>
          <span className="font-medium">{(count ?? 0).toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span>平均金額</span>
          <span className="font-medium">{formatCurrency(avg ?? 0)}</span>
        </div>
      </div>
    </div>
  );
}

/* == 檔期名單卡片（雙11/618）== */
function CampaignCard({
  title,
  rows,
  downloadName,
}: {
  title: string;
  rows: { customer_email: string; amount: number; avgOrder: number; last: Date | string }[];
  downloadName: string;
}) {
  const handleDownload = () => {
    const out = rows.map(r => ({
      customer_email: r.customer_email,
      amount: Math.round(r.amount),
      avg_order: Math.round(r.avgOrder),
      last_order_date: typeof r.last === 'string'
        ? r.last
        : new Date(r.last).toISOString().slice(0, 10),
    }));
    downloadCsv(downloadName, out);
  };

  return (
    <div className="rounded-2xl border border-slate-200 p-4 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
        {rows.length > 0 && (
          <button
            onClick={handleDownload}
            className="px-3 py-1.5 rounded-md border border-slate-300 text-slate-700 hover:bg-slate-50 text-xs"
          >
            下載 CSV
          </button>
        )}
      </div>
      {rows.length === 0 ? (
        <div className="text-slate-400 text-sm text-center py-6">暫無名單</div>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b border-slate-200 text-slate-500">
              <th className="py-2">Email/ID</th>
              <th className="py-2 text-right">檔期間金額</th>
              <th className="py-2 text-right">個人平均</th>
              <th className="py-2 text-right">最後消費(至今)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.customer_email} className="border-b border-slate-100 last:border-none">
                <td className="py-2">{r.customer_email}</td>
                <td className="py-2 text-right">{formatCurrency(r.amount)}</td>
                <td className="py-2 text-right">{formatCurrency(r.avgOrder)}</td>
                <td className="py-2 text-right">
                  {typeof r.last === 'string'
                    ? r.last
                    : new Date(r.last).toISOString().slice(0, 10)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/* == 檔期建議（依關聯規則）== */
function SuggestionCard({
  title,
  rules,
  y,
  winBuilder,
  monthlyAll,
}: {
  title: string;
  rules: { antecedents: string[]; consequents: string[]; support: number; confidence: number; lift: number }[];
  y: number;
  winBuilder: (y: number) => { s: Date; e: Date };
  monthlyAll: { yyyymm: string; revenue: number; orders: number }[];
}) {
  const win = winBuilder(y);
  const s = `${win.s.getFullYear()}-${String(win.s.getMonth() + 1).padStart(2, '0')}-${String(win.s.getDate()).padStart(2, '0')}`;
  const e = `${win.e.getFullYear()}-${String(win.e.getMonth() + 1).padStart(2, '0')}-${String(win.e.getDate()).padStart(2, '0')}`;

  const top = (rules || []).slice(0, 3);

  return (
    <div className="rounded-2xl border border-slate-200 p-4 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
        <span className="text-xs text-slate-500">檔期：{s} ~ {e}</span>
      </div>

      {top.length === 0 ? (
        <div className="text-slate-400 text-sm text-center py-6">尚無顯著規則</div>
      ) : (
        <ol className="list-decimal pl-5 space-y-2 text-sm text-slate-700">
          {top.map((r, i) => (
            <li key={i}>
              <span className="font-medium">{r.antecedents.join(' + ')}</span>
              <span className="mx-1 text-slate-400">→</span>
              <span className="font-medium">{r.consequents.join(' + ')}</span>
              <div className="mt-1 flex gap-2 text-xs text-slate-600">
                <Badge label="Lift" value={r.lift.toFixed(2)} />
                <Badge label="Conf." value={(r.confidence * 100).toFixed(1) + '%'} />
                <Badge label="Supp." value={(r.support * 100).toFixed(1) + '%'} />
              </div>
            </li>
          ))}
        </ol>
      )}

      <p className="mt-3 text-xs text-slate-500">
        建議將以上組合做成套組或加價購，並在檔期前 7~10 天暖身，檔期當週加強再行銷。
      </p>
    </div>
  );
}
