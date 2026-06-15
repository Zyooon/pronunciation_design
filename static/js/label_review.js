document.addEventListener("DOMContentLoaded", () => {
  const filter = document.getElementById("label-review-filter");
  const refresh = document.getElementById("label-review-refresh");

  if (!filter || !refresh) return;

  filter.addEventListener("change", () => loadLabelReviewRows());
  refresh.addEventListener("click", () => loadLabelReviewRows());
  loadLabelReviewRows();
});

async function loadLabelReviewRows() {
  const wrap = document.getElementById("label-review-wrap");
  const filter = document.getElementById("label-review-filter");
  if (!wrap || !filter) return;

  wrap.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';

  try {
    const data = await labelReviewFetchJson(`/api/label-review-results?label=${encodeURIComponent(filter.value)}`);
    if (!data.exists) {
      wrap.innerHTML = `<div class="no-data-msg">data/pronunciation.db 파일이 없습니다.</div>`;
      return;
    }
    if (data.error) {
      wrap.innerHTML = `<div class="no-data-msg">DB 읽기 오류: ${labelReviewEsc(data.error)}</div>`;
      return;
    }
    if (!data.rows || !data.rows.length) {
      wrap.innerHTML = `<div class="no-data-msg">검수할 녹음이 없습니다.</div>`;
      return;
    }

    wrap.innerHTML = `
      <div class="meta-box" style="margin-bottom:12px;">
        <strong>전체:</strong> ${data.row_count}개
        &nbsp;·&nbsp; <strong>표시:</strong> ${data.rows.length}개
        &nbsp;·&nbsp; <strong>필터:</strong> ${labelReviewEsc(filter.value || "전체")}
      </div>
      <div class="table-wrap">
        <table class="data-table" id="label-review-table">
          <thead>
            <tr>
              <th>ID</th><th>날짜</th><th>단어</th><th>발음</th><th>현재 라벨</th>
              <th>점수</th><th>품질/피드백</th><th>오디오</th><th>라벨 지정</th>
            </tr>
          </thead>
          <tbody>
            ${data.rows.map(renderLabelReviewRow).join("")}
          </tbody>
        </table>
      </div>
    `;

    wrap.querySelectorAll("button[data-label]").forEach((btn) => {
      btn.addEventListener("click", () => updateReviewLabel(btn));
    });
  } catch (err) {
    wrap.innerHTML = `<div class="no-data-msg">${labelReviewEsc(err.message)}</div>`;
  }
}

function renderLabelReviewRow(row) {
  const label = row.test_label || "unlabeled";
  const audioHtml = row.recording_path
    ? `<audio controls preload="none" src="/api/recording/${row.id}" style="width:220px;max-width:100%;"></audio>
       <div style="font-size:11px;color:#6b7c8f;max-width:220px;word-break:break-all;">${labelReviewEsc(row.recording_path)}</div>`
    : `<span class="pill pill-warn">파일 없음</span>`;

  const feedbackParts = [];
  if (row.grade) feedbackParts.push(`grade: ${row.grade}`);
  if (row.feedback) feedbackParts.push(row.feedback);
  if (row.total_penalty != null) feedbackParts.push(`penalty: ${row.total_penalty}`);

  return `
    <tr data-recording-id="${row.id}">
      <td class="num-cell">${row.id}</td>
      <td>${labelReviewEsc(row.created_at || "")}</td>
      <td><strong>${labelReviewEsc(row.word || "")}</strong></td>
      <td class="phoneme-cell">/${labelReviewEsc(row.phoneme || "")}/</td>
      <td class="label-cell">${renderLabelPill(label)}</td>
      <td class="num-cell">${row.score ?? "—"}</td>
      <td style="min-width:180px;">${labelReviewEsc(feedbackParts.join(" · ") || "—")}</td>
      <td>${audioHtml}</td>
      <td>${renderLabelButtons(row.id)}</td>
    </tr>
  `;
}

function renderLabelButtons(id) {
  const labels = [
    ["good", "good"],
    ["korean_like", "korean"],
    ["wrong_or_noisy", "wrong"],
    ["exclude", "exclude"],
    ["unlabeled", "clear"],
  ];
  return `<div style="display:flex;flex-wrap:wrap;gap:6px;min-width:180px;">
    ${labels.map(([value, text]) => `<button type="button" class="secondary-btn" data-id="${id}" data-label="${value}" style="padding:6px 8px;font-size:12px;">${text}</button>`).join("")}
  </div>`;
}

function renderLabelPill(label) {
  const cls = label === "good"
    ? "pill-ok"
    : label === "wrong_or_noisy" || label === "exclude"
      ? "pill-err"
      : label === "korean_like"
        ? "pill-warn"
        : "";
  return `<span class="pill ${cls}">${labelReviewEsc(label)}</span>`;
}

async function updateReviewLabel(btn) {
  const id = btn.dataset.id;
  const label = btn.dataset.label;
  const row = btn.closest("tr");
  if (!id || !label || !row) return;

  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "저장 중";

  try {
    const res = await fetch(`/api/user-label/${id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ test_label: label }),
    });
    if (!res.ok) throw new Error(`라벨 저장 실패 — HTTP ${res.status}`);
    const payload = await res.json();
    const nextLabel = payload.test_label || "unlabeled";
    row.querySelector(".label-cell").innerHTML = renderLabelPill(nextLabel);
    row.classList.add("selected");
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

async function labelReviewFetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} — HTTP ${res.status}`);
  return res.json();
}

function labelReviewEsc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
