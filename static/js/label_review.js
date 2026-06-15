const LABEL_REVIEW_PAGE_SIZE = 50;
let _labelReviewPage = 1;
let _labelReviewRowsById = new Map();

// Label Review 탭 전용 스크립트
// - 한 번에 50개씩 서버에서 조회
// - 날짜는 초 단위까지만 표시
// - 피드백은 표시하지 않고 검수에 필요한 핵심 정보만 표시
// - 단어명을 누르면 Word Compare 탭에서 해당 row를 비교한다.
document.addEventListener("DOMContentLoaded", () => {
  const filter = document.getElementById("label-review-filter");
  const refresh = document.getElementById("label-review-refresh");

  if (!filter || !refresh) return;

  filter.addEventListener("change", () => {
    _labelReviewPage = 1;
    loadLabelReviewRows();
  });
  refresh.addEventListener("click", () => loadLabelReviewRows());
  loadLabelReviewRows();
});

async function loadLabelReviewRows(page = _labelReviewPage) {
  const wrap = document.getElementById("label-review-wrap");
  const filter = document.getElementById("label-review-filter");
  if (!wrap || !filter) return;

  _labelReviewPage = Math.max(1, page);
  wrap.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';

  try {
    const params = new URLSearchParams({
      label: filter.value,
      page: String(_labelReviewPage),
      limit: String(LABEL_REVIEW_PAGE_SIZE),
    });
    const data = await labelReviewFetchJson(`/api/label-review-results?${params.toString()}`);
    _labelReviewPage = data.page || _labelReviewPage;
    _labelReviewRowsById = new Map((data.rows || []).map((row) => [String(row.id), row]));

    if (!data.exists) {
      wrap.innerHTML = `<div class="no-data-msg">data/pronunciation.db 파일이 없습니다.</div>`;
      return;
    }
    if (data.error) {
      wrap.innerHTML = `<div class="no-data-msg">DB 읽기 오류: ${labelReviewEsc(data.error)}</div>`;
      return;
    }
    if (!data.rows || !data.rows.length) {
      wrap.innerHTML = `
        ${renderLabelReviewPager(data)}
        <div class="no-data-msg">검수할 녹음이 없습니다.</div>
      `;
      bindLabelReviewPager(wrap, data);
      return;
    }

    const start = (data.page - 1) * data.page_size + 1;
    const end = start + data.rows.length - 1;

    wrap.innerHTML = `
      <div class="meta-box label-review-meta">
        <strong>전체:</strong> ${data.row_count}개
        &nbsp;·&nbsp; <strong>표시:</strong> ${start}-${end}개
        &nbsp;·&nbsp; <strong>페이지:</strong> ${data.page}/${data.total_pages || 1}
        &nbsp;·&nbsp; <strong>필터:</strong> ${labelReviewEsc(filter.value || "전체")}
      </div>
      ${renderLabelReviewPager(data)}
      <div id="label-review-message" class="meta-box" style="display:none;margin-bottom:12px;"></div>
      <div class="table-wrap label-review-table-wrap">
        <table class="data-table label-review-table" id="label-review-table">
          <thead>
            <tr>
              <th>ID</th><th>날짜</th><th>단어</th><th>발음</th><th>현재 라벨</th>
              <th>점수</th><th>Penalty</th><th>오디오</th><th>라벨 지정</th>
            </tr>
          </thead>
          <tbody>
            ${data.rows.map(renderLabelReviewRow).join("")}
          </tbody>
        </table>
      </div>
      ${renderLabelReviewPager(data)}
    `;

    wrap.querySelectorAll("button[data-label]").forEach((btn) => {
      btn.addEventListener("click", () => updateReviewLabel(btn));
    });
    wrap.querySelectorAll("button[data-compare-id]").forEach((btn) => {
      btn.addEventListener("click", () => openCompareFromLabelReview(btn));
    });
    bindLabelReviewPager(wrap, data);
  } catch (err) {
    wrap.innerHTML = `<div class="no-data-msg">${labelReviewEsc(err.message)}</div>`;
  }
}

function renderLabelReviewRow(row) {
  const label = row.test_label || "unlabeled";
  const audioHtml = row.recording_path
    ? `<audio class="label-review-audio" controls preload="none" src="/api/recording/${row.id}"></audio>
       <div class="label-review-path">${labelReviewEsc(row.recording_path)}</div>`
    : `<span class="pill pill-warn">파일 없음</span>`;

  return `
    <tr data-recording-id="${row.id}">
      <td class="num-cell" data-label="ID">${row.id}</td>
      <td data-label="날짜">${labelReviewEsc(formatLabelReviewDate(row.created_at))}</td>
      <td data-label="단어">
        <button type="button" class="word-compare-link" data-compare-id="${row.id}" title="Word Compare에서 보기">
          ${labelReviewEsc(row.word || "")}
        </button>
      </td>
      <td class="phoneme-cell" data-label="발음">/${labelReviewEsc(row.phoneme || "")}/</td>
      <td class="label-cell" data-label="현재 라벨">${renderLabelPill(label)}</td>
      <td class="num-cell" data-label="점수">${row.score ?? "—"}</td>
      <td class="num-cell" data-label="Penalty">${row.total_penalty ?? "—"}</td>
      <td data-label="오디오">${audioHtml}</td>
      <td data-label="라벨 지정">${renderLabelButtons(row.id)}</td>
    </tr>
  `;
}

function openCompareFromLabelReview(btn) {
  const id = String(btn.dataset.compareId || "");
  const row = _labelReviewRowsById.get(id);
  if (!row) {
    alert("해당 녹음 row를 찾을 수 없습니다. 새로고침 후 다시 시도해주세요.");
    return;
  }
  if (typeof window.openWordCompareFromLabelReview !== "function") {
    alert("Word Compare 연결 함수를 찾을 수 없습니다. 페이지를 새로고침해주세요.");
    return;
  }
  window.openWordCompareFromLabelReview(row);
}

function renderLabelReviewPager(data) {
  const totalPages = data.total_pages || 0;
  if (totalPages <= 1) return "";
  const page = data.page || 1;
  return `
    <div class="label-review-pager">
      <button type="button" class="secondary-btn" data-page="1" ${page <= 1 ? "disabled" : ""}>처음</button>
      <button type="button" class="secondary-btn" data-page="${page - 1}" ${page <= 1 ? "disabled" : ""}>이전</button>
      <span class="pager-status">${page} / ${totalPages}</span>
      <button type="button" class="secondary-btn" data-page="${page + 1}" ${page >= totalPages ? "disabled" : ""}>다음</button>
      <button type="button" class="secondary-btn" data-page="${totalPages}" ${page >= totalPages ? "disabled" : ""}>마지막</button>
    </div>
  `;
}

function bindLabelReviewPager(wrap, data) {
  const totalPages = data.total_pages || 0;
  wrap.querySelectorAll(".label-review-pager button[data-page]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const nextPage = Math.min(Math.max(parseInt(btn.dataset.page, 10), 1), totalPages || 1);
      loadLabelReviewRows(nextPage);
    });
  });
}

function renderLabelButtons(id) {
  const labels = [
    ["good", "good"],
    ["korean_like", "korean"],
    ["wrong_or_noisy", "wrong"],
    ["exclude", "exclude"],
    ["unlabeled", "clear"],
  ];
  return `<div class="label-review-actions">
    ${labels.map(([value, text]) => `<button type="button" class="secondary-btn" data-id="${id}" data-label="${value}">${text}</button>`).join("")}
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
  clearLabelReviewMessage();

  try {
    const payload = await postReviewLabel(id, label);
    const nextLabel = payload.test_label || "unlabeled";
    row.querySelector(".label-cell").innerHTML = renderLabelPill(nextLabel);
    row.classList.add("selected");
    const cachedRow = _labelReviewRowsById.get(String(id));
    if (cachedRow) cachedRow.test_label = payload.test_label;
    showLabelReviewMessage(`ID ${id} 라벨이 ${nextLabel}(으)로 저장됐습니다.`, "ok");

    const currentFilter = document.getElementById("label-review-filter")?.value || "";
    if (currentFilter && currentFilter !== nextLabel) {
      setTimeout(() => loadLabelReviewRows(_labelReviewPage), 350);
    }
  } catch (err) {
    showLabelReviewMessage(err.message, "error");
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

async function postReviewLabel(id, label) {
  const res = await fetch(`/api/user-label/${id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "application/json" },
    body: JSON.stringify({ test_label: label }),
  });
  let payload = null;
  try {
    payload = await res.json();
  } catch (_) {
    payload = null;
  }
  if (!res.ok) {
    const detail = payload?.detail || payload?.error || `HTTP ${res.status}`;
    throw new Error(`라벨 저장 실패: ${detail}`);
  }
  if (!payload?.ok) {
    throw new Error("라벨 저장 실패: 서버 응답이 올바르지 않습니다.");
  }
  return payload;
}

async function labelReviewFetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} — HTTP ${res.status}`);
  return res.json();
}

function showLabelReviewMessage(message, type = "ok") {
  const box = document.getElementById("label-review-message");
  if (!box) return;
  box.style.display = "block";
  box.style.borderColor = type === "error" ? "#fecaca" : "#bbf7d0";
  box.style.color = type === "error" ? "#991b1b" : "#065f46";
  box.style.background = type === "error" ? "#fef2f2" : "#f0fdf4";
  box.textContent = message;
}

function clearLabelReviewMessage() {
  const box = document.getElementById("label-review-message");
  if (!box) return;
  box.style.display = "none";
  box.textContent = "";
}

function formatLabelReviewDate(value) {
  if (!value) return "";
  const raw = String(value);
  const match = raw.match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})/);
  if (match) return `${match[1]} ${match[2]}`;
  return raw.length > 19 ? raw.slice(0, 19).replace("T", " ") : raw.replace("T", " ");
}

function labelReviewEsc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
