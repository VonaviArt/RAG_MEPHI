import "./style.css";

type Chunk = {
  text: string;
  score: number;
  metadata: Record<string, unknown>;
};

type QueryResponse = {
  answer: string;
  chunks: Chunk[];
};

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("App root not found");
}

app.innerHTML = `
  <main class="container">
    <header class="page-header">
      <div class="brand">
        <p class="brand-kicker">Ассистент</p>
        <h1 class="page-title">RAG MEPHI</h1>
      </div>
    </header>

    <section class="section section--query" aria-labelledby="question-heading">
      <h2 id="question-heading" class="section-title">Вопрос</h2>
      <form id="queryForm" class="query-form">
        <textarea id="question" rows="4" placeholder="Привет! Чем могу помочь? (Enter — отправить, Shift+Enter — новая строка)" autocomplete="off"></textarea>
        <div class="actions">
          <button id="submitBtn" type="submit" class="btn-submit">
            <span class="btn-spinner" aria-hidden="true"></span>
            <span class="btn-text">Отправить</span>
          </button>
        </div>
      </form>
    </section>

    <section class="section" aria-labelledby="answer-heading">
      <h2 id="answer-heading" class="section-title">Ответ</h2>
      <div id="answer" class="panel panel--muted">Ответ появится здесь после отправки вопроса.</div>
    </section>

    <section class="section section--context" aria-labelledby="chunks-heading">
      <h2 id="chunks-heading" class="section-title">Контекст</h2>
      <div id="chunks" class="chunks"></div>
    </section>

    <footer class="section section--contacts" aria-labelledby="contacts-heading">
      <h2 id="contacts-heading" class="section-title">Контакты</h2>
      <address class="panel contacts-panel">
        <p class="contacts-line">
          <span class="contacts-key">Телефон:</span>
          <a href="tel:+74957885699">+7 (495) 788-56-99</a>; <a href="tel:+74993247777">+7 (499) 324-77-77</a>
        </p>
        <p class="contacts-line">
          <span class="contacts-key">Телефонный справочник НИЯУ МИФИ:</span>
          <a href="https://voip.mephi.ru" target="_blank" rel="noopener noreferrer">voip.mephi.ru</a>
        </p>
        <p class="contacts-line">
          <span class="contacts-key">Электронная почта:</span>
          <a href="mailto:info@mephi.ru">info@mephi.ru</a>
        </p>
        <p class="contacts-line">
          <span class="contacts-key">Электронная почта для абитуриентов:</span>
          <a href="mailto:school@mephi.ru">school@mephi.ru</a>
        </p>
      </address>
    </footer>
  </main>
`;

const form = document.querySelector("#queryForm");
const questionField = document.querySelector("#question");
const submitButton = document.querySelector("#submitBtn");
const answerEl = document.querySelector("#answer");
const chunksBlock = document.querySelector("#chunks");

if (
  !(form instanceof HTMLFormElement) ||
  !(questionField instanceof HTMLTextAreaElement) ||
  !(submitButton instanceof HTMLButtonElement) ||
  !(answerEl instanceof HTMLDivElement) ||
  !(chunksBlock instanceof HTMLDivElement)
) {
  throw new Error("Required UI elements not found");
}

const labelInBtn = submitButton.querySelector(".btn-text");
if (!(labelInBtn instanceof HTMLSpanElement)) {
  throw new Error("Button markup incomplete");
}

const chunksRoot = chunksBlock;

mountApp({
  form,
  questionField,
  submitButton,
  labelInBtn,
  answerEl,
  chunksRoot,
});

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

type UiRefs = {
  form: HTMLFormElement;
  questionField: HTMLTextAreaElement;
  submitButton: HTMLButtonElement;
  labelInBtn: HTMLSpanElement;
  answerEl: HTMLDivElement;
  chunksRoot: HTMLDivElement;
};

function mountApp(refs: UiRefs): void {
  const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
  const {
    form,
    questionField,
    submitButton,
    labelInBtn,
    answerEl,
    chunksRoot,
  } = refs;

  function setLoading(loading: boolean): void {
    submitButton.disabled = loading;
    submitButton.classList.toggle("btn-submit--busy", loading);
    labelInBtn.textContent = loading ? "Обрабатываю…" : "Отправить";
  }

  function renderChunks(chunks: Chunk[]): void {
    if (!chunks.length) {
      chunksRoot.innerHTML = `<div class="panel panel--muted">Контекстные чанки не найдены.</div>`;
      return;
    }

    chunksRoot.innerHTML = chunks
      .map((chunk, index) => {
        const metadata = escapeHtml(JSON.stringify(chunk.metadata));
        const body = escapeHtml(chunk.text);
        return `
        <article class="panel chunk">
          <div class="chunk-header">
            <span class="chunk-label">Фрагмент ${index + 1}</span>
            <span class="chunk-score">${chunk.score.toFixed(4)}</span>
          </div>
          <p class="chunk-body">${body}</p>
          <div class="meta">${metadata}</div>
        </article>
      `;
      })
      .join("");
  }

  async function runQuery(): Promise<void> {
    const question = questionField.value.trim();
    if (!question) {
      answerEl.textContent = "Введите вопрос.";
      answerEl.classList.add("panel--muted");
      return;
    }

    setLoading(true);
    answerEl.textContent = "Думаем над ответом…";
    answerEl.classList.add("panel--muted");
    chunksRoot.innerHTML = "";
    answerEl.scrollIntoView({ behavior: "smooth", block: "start" });

    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Request failed with status ${response.status}`);
      }

      const data = (await response.json()) as QueryResponse;
      answerEl.textContent = data.answer || "Ответ пустой.";
      answerEl.classList.remove("panel--muted");
      renderChunks(data.chunks || []);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Неизвестная ошибка";
      answerEl.textContent = `Ошибка: ${message}`;
      answerEl.classList.add("panel--muted");
      chunksRoot.innerHTML = "";
    } finally {
      setLoading(false);
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void runQuery();
  });

  questionField.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    void runQuery();
  });
}
