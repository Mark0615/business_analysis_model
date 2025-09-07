'use client';
export default function Error({ error }: { error: Error }) {
  return (
    <div className="p-10 text-center">
      <h1 className="text-2xl font-bold mb-2">發生錯誤</h1>
      <p className="text-slate-500">{error.message}</p>
      <p className="text-slate-400 mt-2">請稍候再試或重新整理頁面。</p>
    </div>
  );
}
