const records = [
  { time: '2026-03-12 09:20:11', name: '张伟', camera: 'A-01', score: 0.95, state: '正常' },
  { time: '2026-03-12 09:18:42', name: '未知人员', camera: 'B-03', score: 0.61, state: '告警' },
  { time: '2026-03-12 09:15:35', name: '李敏', camera: 'A-01', score: 0.91, state: '正常' },
  { time: '2026-03-12 09:12:06', name: '王芳', camera: 'B-03', score: 0.87, state: '正常' },
  { time: '2026-03-12 09:10:27', name: '未知人员', camera: 'A-01', score: 0.58, state: '告警' },
  { time: '2026-03-12 09:08:03', name: '赵强', camera: 'B-03', score: 0.89, state: '正常' }
];

let page = 1;
const pageSize = 4;
let desc = true;

const recordBody = document.getElementById('recordBody');
const pageInfo = document.getElementById('pageInfo');
const sortBtn = document.getElementById('sortTime');
const cameraFilter = document.getElementById('cameraFilter');
const keywordInput = document.getElementById('keyword');
const statusText = document.getElementById('statusText');

function filteredData() {
  const camera = cameraFilter.value;
  const keyword = keywordInput.value.trim();
  return records.filter(r => {
    const camOk = camera === 'all' || r.camera === camera;
    const kwOk = !keyword || r.name.includes(keyword) || r.camera.includes(keyword);
    return camOk && kwOk;
  });
}

function renderTable() {
  const data = filteredData().sort((a, b) => desc ? b.time.localeCompare(a.time) : a.time.localeCompare(b.time));
  const total = Math.max(1, Math.ceil(data.length / pageSize));
  if (page > total) page = total;
  const start = (page - 1) * pageSize;
  const rows = data.slice(start, start + pageSize);

  recordBody.innerHTML = rows.map(row => `
    <tr>
      <td>${row.time}</td>
      <td>${row.name}</td>
      <td>${row.camera}</td>
      <td>${(row.score * 100).toFixed(1)}%</td>
      <td class="${row.state === '正常' ? 'tag-ok' : 'tag-warn'}">${row.state}</td>
    </tr>
  `).join('') || '<tr><td colspan="5">暂无数据</td></tr>';

  pageInfo.textContent = `第 ${page} / ${total} 页`;
}

sortBtn.addEventListener('click', () => {
  desc = !desc;
  renderTable();
});

cameraFilter.addEventListener('change', () => {
  page = 1;
  renderTable();
});
keywordInput.addEventListener('input', () => {
  page = 1;
  renderTable();
});

document.getElementById('prevPage').addEventListener('click', () => {
  if (page > 1) page -= 1;
  renderTable();
});

document.getElementById('nextPage').addEventListener('click', () => {
  page += 1;
  renderTable();
});

const dialog = document.getElementById('confirmDialog');
document.getElementById('exportBtn').addEventListener('click', () => dialog.showModal());
document.getElementById('cancelExport').addEventListener('click', () => dialog.close());
document.getElementById('confirmExport').addEventListener('click', () => {
  dialog.close();
  statusText.textContent = '导出任务已提交，系统正在后台生成文件。';
});

document.getElementById('refreshBtn').addEventListener('click', () => {
  statusText.textContent = '数据已刷新，最后更新时间：' + new Date().toLocaleTimeString('zh-CN', { hour12: false });
});

document.getElementById('clearNotice').addEventListener('click', () => {
  statusText.textContent = '系统运行正常，暂无未处理错误。';
});

renderTable();
